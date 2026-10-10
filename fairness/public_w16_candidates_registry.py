"""Generalized candidate-level objection logger.

Reuses the Australian W16 ML core (numpy W-BCE + two raw-scale plug-in
regressions, validation-frozen criterion thresholds) but takes any registered dataset that
yields (features, label, value_A, loss_B, group). Emits the VERBOSE candidate
schema that `summarize_w16_objection_penalty.py` already accepts, so the same
objection/grievance postprocessor runs unchanged per dataset.

Convention: A=(1-y)*value on the good branch, B=amount on the bad branch;
ratio target E[B|x]/E[A|x]. Group excluded from features (group-blind base).
Objecting norm set downstream is {harm, net_harm}; wbce_ratio is the baseline
diagnostic (OI-W16-F).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

CRITERIA = ("wbce_ratio", "harm", "net_harm")
VERBOSE_FIELDS = (
    "dataset", "family", "seed", "target", "sample_split", "group",
    "value", "loss", "estimated_value", "estimated_loss",
    "selected_criterion_wbce_ratio", "selected_criterion_harm",
    "selected_criterion_net_harm", "account",
)


# ---------------- ML core (ported verbatim from icaif_fair_w16_candidates) ----------------
def sigmoid(v):
    out = np.empty_like(v, dtype=np.float64)
    p = v >= 0
    out[p] = 1.0 / (1.0 + np.exp(-v[p]))
    e = np.exp(v[~p]); out[~p] = e / (1.0 + e)
    return out


def _adam_logistic(x, y, w, steps, seed):
    rng = np.random.default_rng(seed)
    x = x.astype(np.float64); y = y.astype(np.float64); w = w.astype(np.float64)
    ws = max(float(w.sum()), 1e-12)
    coef = rng.normal(0, 1e-3, size=x.shape[1]); b = 0.0
    cm = np.zeros_like(coef); cv = np.zeros_like(coef); bm = bv = 0.0
    b1, b2, lr, l2 = 0.9, 0.999, 0.05, 1e-4
    for t in range(1, steps + 1):
        err = sigmoid(x @ coef + b) - y
        we = w * err / ws
        gc = x.T @ we + l2 * coef; gb = float(we.sum())
        cm = b1 * cm + (1 - b1) * gc; cv = b2 * cv + (1 - b2) * gc * gc
        bm = b1 * bm + (1 - b1) * gb; bv = b2 * bv + (1 - b2) * gb * gb
        coef -= lr * (cm / (1 - b1**t)) / (np.sqrt(cv / (1 - b2**t)) + 1e-8)
        b -= lr * (bm / (1 - b1**t)) / (math.sqrt(bv / (1 - b2**t)) + 1e-8)
    return coef, b


def fit_wbce(x, loss, value, seed, steps):
    pos = loss > 0; neg = value > 0
    xf = np.concatenate([x[pos], x[neg]])
    yf = np.concatenate([np.ones(int(pos.sum())), np.zeros(int(neg.sum()))])
    w = np.concatenate([loss[pos], value[neg]]).astype(np.float64)
    w = w / max(float(w.mean()), 1e-12)
    return _adam_logistic(xf, yf, w, steps, seed)


def fit_raw_mean(x, y, steps, seed):
    rng = np.random.default_rng(seed)
    x = x.astype(np.float64)
    tgt = np.maximum(y.astype(np.float64), 0.0)
    loc = float(tgt.mean()) if len(tgt) else 0.0
    scale_y = float(tgt.std(ddof=0)) if len(tgt) else 1.0
    if scale_y <= 1e-12:
        scale_y = 1.0
    tgt = (tgt - loc) / scale_y
    coef = rng.normal(0, 1e-3, size=x.shape[1]); b = float(tgt.mean()) if len(tgt) else 0.0
    cm = np.zeros_like(coef); cv = np.zeros_like(coef); bm = bv = 0.0
    b1, b2, lr, l2 = 0.9, 0.999, 0.03, 1e-4
    scale = max(float(len(tgt)), 1.0)
    for t in range(1, steps + 1):
        res = x @ coef + b - tgt
        gc = (x.T @ res) / scale + l2 * coef; gb = float(res.mean()) if len(res) else 0.0
        cm = b1 * cm + (1 - b1) * gc; cv = b2 * cv + (1 - b2) * gc * gc
        bm = b1 * bm + (1 - b1) * gb; bv = b2 * bv + (1 - b2) * gb * gb
        coef -= lr * (cm / (1 - b1**t)) / (np.sqrt(cv / (1 - b2**t)) + 1e-8)
        b -= lr * (bm / (1 - b1**t)) / (math.sqrt(bv / (1 - b2**t)) + 1e-8)
    return coef, b, loc, scale_y


def predict_raw_mean(x, coef, b, loc, scale_y):
    return np.maximum((x.astype(np.float64) @ coef + b) * scale_y + loc, 0.0)


def threshold_for_mass(scores, mass, frac):
    order = np.argsort(-scores, kind="stable")
    cum = np.cumsum(mass[order], dtype=np.float64)
    if cum[-1] <= 0:
        return float("inf")
    pos = min(int(np.searchsorted(cum, frac * cum[-1], side="left")), len(order) - 1)
    return float(scores[order[pos]])


def stratified_split(idx, labels, frac, seed):
    rng = np.random.default_rng(seed); left, right = [], []
    for lab in np.unique(labels[idx]):
        cell = idx[labels[idx] == lab].copy(); rng.shuffle(cell)
        k = 0 if len(cell) <= 1 else min(max(int(round(frac * len(cell))), 1), len(cell) - 1)
        right.append(cell[:k]); left.append(cell[k:])
    l = np.concatenate(left); r = np.concatenate(right); rng.shuffle(l); rng.shuffle(r)
    return l, r


# ---------------- feature transform (manual, train-fit, no sklearn) ----------------
def transform(frame, numeric, categorical, tr, va, te):
    num = frame[numeric].apply(pd.to_numeric, errors="coerce")
    num = num.fillna(num.iloc[tr].median()).fillna(0.0)
    tn = num.iloc[tr].to_numpy(np.float64)
    mean = tn.mean(0); std = tn.std(0); std[std <= 0] = 1.0
    cats = frame[categorical].fillna("NA").astype(str)
    levels = {c: sorted(cats.iloc[tr][c].unique()) for c in categorical}

    def build(ix):
        parts = [((num.iloc[ix].to_numpy(np.float64) - mean) / std)]
        for c in categorical:
            lv = {v: i for i, v in enumerate(levels[c])}
            oh = np.zeros((len(ix), len(lv)), np.float64)
            for r, v in enumerate(cats.iloc[ix][c].to_numpy()):
                if v in lv:
                    oh[r, lv[v]] = 1.0
            parts.append(oh)
        return np.concatenate(parts, 1).astype(np.float64)
    return build(tr), build(va), build(te)


# ---------------- dataset registry ----------------
def load_belgian(path, n, seed):
    import rdata
    obj = rdata.read_rda(path)
    frame = pd.DataFrame(obj["beMTPL97"])
    amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(0).to_numpy(np.float64)
    expo = pd.to_numeric(frame["expo"], errors="coerce").fillna(0).to_numpy(np.float64)
    sex = frame["sex"].astype(str).to_numpy()
    valid = expo > 0
    frame, amount, expo, sex = frame.loc[valid].reset_index(drop=True), amount[valid], expo[valid], sex[valid]
    label = (amount > 0).astype(np.float64)
    if n and n < len(frame):
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(frame), n, replace=False)
        frame, amount, expo, sex, label = frame.iloc[keep].reset_index(drop=True), amount[keep], expo[keep], sex[keep], label[keep]
    value_A = expo * (1.0 - label)   # good branch
    loss_B = amount                  # bad branch amount
    numeric = ["expo", "ageph", "bm", "power", "agec", "long", "lat"]
    categorical = ["coverage", "fuel", "use", "fleet"]   # sex excluded from features
    return frame, numeric, categorical, label, value_A, loss_B, sex, "beMTPL97", None


CHRONIC = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]
DIAG = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC = [f"ClmProcedureCode_{i}" for i in range(1, 7)]


def account_nested_split(account, label, seed):
    """Account-disjoint train/val/test (64/16/20 of accounts)."""
    accts = np.unique(account)
    order = accts.copy()
    np.random.default_rng(seed).shuffle(order)
    n = len(order); n_test = int(0.20 * n); n_val = int(0.20 * (n - n_test))
    where = {}
    for a in order[:n_test]: where[a] = 2
    for a in order[n_test:n_test + n_val]: where[a] = 1
    for a in order[n_test + n_val:]: where[a] = 0
    tag = np.array([where[a] for a in account])
    return np.where(tag == 0)[0], np.where(tag == 1)[0], np.where(tag == 2)[0]


def _load_cms(path, n, seed, attr):
    use = set(["Provider", "PotentialFraud", "InscClaimAmtReimbursed", "Gender", "Race",
               "Age", "ClaimStartDt", "ClaimEndDt", "AdmissionDt", "AdmitForDays",
               "RenalDiseaseIndicator", "NoOfMonths_PartACov", "NoOfMonths_PartBCov",
               "DeductibleAmtPaid", "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
               "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"] + CHRONIC + DIAG + PROC)
    df = pd.read_csv(path, usecols=lambda c: c in use, low_memory=False)
    amount = pd.to_numeric(df["InscClaimAmtReimbursed"], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    label = df["PotentialFraud"].eq("Yes").to_numpy(np.float64)
    valid = amount > 0
    df = df.loc[valid].reset_index(drop=True); amount = amount[valid]; label = label[valid]
    loss_B = amount * label
    value_A = amount * (1.0 - label)
    start = pd.to_datetime(df["ClaimStartDt"], errors="coerce"); end = pd.to_datetime(df["ClaimEndDt"], errors="coerce")
    feats = pd.DataFrame({
        "age": pd.to_numeric(df["Age"], errors="coerce"),
        "claim_days": (end - start).dt.days.clip(lower=0),
        "admit_days": pd.to_numeric(df["AdmitForDays"], errors="coerce").fillna(0),
        "is_inpatient": df["AdmissionDt"].notna().astype(float),
        "renal": pd.to_numeric(df["RenalDiseaseIndicator"].astype(str).replace("Y", "1"), errors="coerce").fillna(0),
        "part_a": pd.to_numeric(df["NoOfMonths_PartACov"], errors="coerce"),
        "part_b": pd.to_numeric(df["NoOfMonths_PartBCov"], errors="coerce"),
        "deductible": pd.to_numeric(df["DeductibleAmtPaid"], errors="coerce").fillna(0),
        "ip_reimb": pd.to_numeric(df["IPAnnualReimbursementAmt"], errors="coerce").fillna(0),
        "op_reimb": pd.to_numeric(df["OPAnnualReimbursementAmt"], errors="coerce").fillna(0),
        "diag_count": df[DIAG].notna().sum(axis=1),
        "proc_count": df[PROC].notna().sum(axis=1),
    })
    for c in CHRONIC:
        feats[c] = pd.to_numeric(df[c], errors="coerce")
    if attr == "gender":
        group = df["Gender"].astype("Int64").astype(str).to_numpy()
    else:
        race = df["Race"].astype("Int64").astype(str).to_numpy()
        group = np.where(race == "1", "1_ref", "2_other")  # white(1) vs rest, binary gap
    account = df["Provider"].astype(str).to_numpy()
    if n and len(feats) > n:
        rng = np.random.default_rng(seed); provs = np.unique(account)
        keepp = set(rng.choice(provs, max(2, int(len(provs) * n / len(feats))), replace=False))
        keep = np.array([a in keepp for a in account])
        feats = feats.loc[keep].reset_index(drop=True)
        label, loss_B, value_A, group, account = label[keep], loss_B[keep], value_A[keep], group[keep], account[keep]
    return feats, list(feats.columns), [], label, value_A, loss_B, group, f"cms_{attr}", account


GENDER_PATS = ["sex", "gender", "male", "kon"]
EXPO_PATS = ["yearexp", "exposure", "expo", "duration", "insured", "poldur"]
AMOUNT_PATS = ["claimamount", "claimcost", "claimcst", "claimtotal", "claimpaid",
               "paidloss", "payment", "severity", "losscost", "amount", "cost", "loss"]
AGE_PATS = ["drivage", "ageph", "licage", "ownerage", "driverage", "polholderage", "age"]
EXCL_FEAT = ["claim", "nclaim", "numclaim", "freq", "average", "payment", "paid",
             "severity", "sev", "losscost", "cost", "policyid", "polid", "idpol",
             "policynum", "postcode", "recordbeg", "recordend"]


def _norm(c):
    import re
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def _find(cols, pats, exclude=()):
    nc = {c: _norm(c) for c in cols}
    for p in pats:
        for c in cols:
            if p in nc[c] and not any(e in nc[c] for e in exclude):
                return c
    return None


def generic_load(path, n, seed):
    """Auto-detect (exposure=A, amount=B, gender/age group) from any CAS .rda."""
    import os
    import rdata
    obj = rdata.read_rda(path)
    best = None
    for k, v in obj.items():
        try:
            df = pd.DataFrame(v)
        except Exception:
            continue
        if len(df) and (best is None or len(df) > len(best[1])):
            best = (k, df)
    if best is None:
        raise RuntimeError("no dataframe in rda")
    df = best[1]; df.columns = [str(c) for c in df.columns]; cols = list(df.columns)
    expo_c = _find(cols, EXPO_PATS)
    amt_c = _find(cols, AMOUNT_PATS, exclude=["average", "nclaim", "claimnb",
                                             "claimnumber", "claimcount", "claimind", "claimfreq"])
    if not expo_c or not amt_c or expo_c == amt_c:
        raise RuntimeError(f"missing expo/amount (expo={expo_c}, amt={amt_c})")
    amount = pd.to_numeric(df[amt_c], errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float64)
    expo = pd.to_numeric(df[expo_c], errors="coerce").fillna(0).to_numpy(np.float64)
    valid = expo > 0
    df = df.loc[valid].reset_index(drop=True); amount = amount[valid]; expo = expo[valid]
    label = (amount > 0).astype(np.float64)
    if label.sum() < 30 or (1 - label).sum() < 30:
        raise RuntimeError(f"degenerate labels (pos={int(label.sum())})")
    value_A = expo * (1.0 - label); loss_B = amount
    gcol = _find(cols, GENDER_PATS); acol = _find(cols, AGE_PATS)
    gsrc = None
    if gcol:
        g = df[gcol].astype(str).to_numpy(); gattr = gcol; gsrc = gcol
    elif acol:
        a = pd.to_numeric(df[acol], errors="coerce"); med = float(a.median())
        g = np.where(a.to_numpy() <= med, "young", "old"); gattr = f"{acol}_bin"; gsrc = acol
    else:
        g = np.array(["all"] * len(df)); gattr = "none"
    uniq, counts = np.unique(g, return_counts=True)
    if len(uniq) > 2:
        top = uniq[int(np.argmax(counts))]; g = np.where(g == top, str(top), "other")
    numeric, categorical = [], []
    for c in cols:
        if c in (gsrc,): continue
        nn = _norm(c)
        if c != expo_c and any(e in nn for e in EXCL_FEAT): continue
        if c == amt_c: continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() > 0.8:
            numeric.append(c)
        elif df[c].astype(str).nunique() <= 20:
            categorical.append(c)
    if not numeric and not categorical:
        raise RuntimeError("no usable features")
    frame = df[list(dict.fromkeys(numeric + categorical))].copy()
    dsname = os.path.basename(path).replace(".rda", "")
    if n and len(frame) > n:
        rng = np.random.default_rng(seed); idx = rng.choice(len(frame), n, replace=False)
        frame = frame.iloc[idx].reset_index(drop=True)
        label, value_A, loss_B, g = label[idx], value_A[idx], loss_B[idx], g[idx]
    return frame, numeric, categorical, label, value_A, loss_B, g, f"{dsname}[{gattr}]", None


REGISTRY = {
    "belgian": load_belgian,
    "cms_gender": lambda p, n, s: _load_cms(p, n, s, "gender"),
    "cms_race": lambda p, n, s: _load_cms(p, n, s, "race"),
    "auto": generic_load,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(REGISTRY))
    p.add_argument("--input", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n", type=int, default=60000)
    p.add_argument("--target", type=float, default=0.10)
    p.add_argument("--net-lambda", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    frame, numeric, categorical, label, value_A, loss_B, group, dsname, account = REGISTRY[args.dataset](
        args.input, args.n, 42
    )
    print(f"{dsname}: n={len(frame)} pos_rate={label.mean():.4f} "
          f"account_disjoint={account is not None} "
          f"groups={dict(zip(*np.unique(group, return_counts=True)))}", flush=True)
    rows = []
    idx = np.arange(len(frame))
    for seed in range(args.seeds):
        if account is None:
            outer, te = stratified_split(idx, label, 0.20, seed)
            tr, va = stratified_split(outer, label, 0.20, 10_000 + seed)
        else:
            tr, va, te = account_nested_split(account, label, seed)
        xtr, xva, xte = transform(frame, numeric, categorical, tr, va, te)
        wc, wb = fit_wbce(xtr, loss_B[tr], value_A[tr], 500_000 + seed, args.steps)
        vc, vb, vl, vs = fit_raw_mean(xtr, value_A[tr], args.steps, seed + 1)
        lc, lb, ll, ls = fit_raw_mean(xtr, loss_B[tr], args.steps, seed + 2)
        for split, ix, xe in (("validation", va, xva), ("test", te, xte)):
            score = xe @ wc + wb
            ah = np.maximum(predict_raw_mean(xe, vc, vb, vl, vs), 1e-9)
            bh = predict_raw_mean(xe, lc, lb, ll, ls)
            crit = {
                "wbce_ratio": score,
                "harm": bh,
                "net_harm": bh - args.net_lambda * ah,
            }
            # thresholds frozen on validation, at the value budget
            if split == "validation":
                thr = {c: threshold_for_mass(crit[c], value_A[ix], args.target) for c in CRITERIA}
                frozen_thr = thr
            selected = {c: (crit[c] >= frozen_thr[c]).astype(int) for c in CRITERIA}
            for j in range(len(ix)):
                rows.append({
                    "dataset": dsname, "family": "natural", "seed": seed,
                    "target": args.target, "sample_split": split, "group": group[ix][j],
                    "value": f"{value_A[ix][j]:.10g}", "loss": f"{loss_B[ix][j]:.10g}",
                    "estimated_value": f"{ah[j]:.10g}", "estimated_loss": f"{bh[j]:.10g}",
                    "selected_criterion_wbce_ratio": selected["wbce_ratio"][j],
                    "selected_criterion_harm": selected["harm"][j],
                    "selected_criterion_net_harm": selected["net_harm"][j],
                    "account": (str(account[ix][j]) if account is not None else ""),
                })
        print(f"  seed {seed} done", flush=True)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=VERBOSE_FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"saved {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
