# Yielding Fairness

Public repository: [atavory/yielding_fairness](https://github.com/atavory/yielding_fairness).

This repository contains public Python runners for the value-weighted blocking
experiments.  It intentionally excludes raw datasets, frozen result CSVs,
aggregation scripts, and plotting scripts.  Those artifacts live in the
submission evidence bundle.

The package includes:

- `score/`: public score-learning runners for the ten yield-Gini datasets.
- `fairness/export_cas_rda_to_csv.py`: converts CASdatasets `.rda` data frames to
  CSV without requiring an R runtime.
- `fairness/fairness_candidates.py`: generates candidate-level fairness logs for Australian
  MTPL, Belgian MTPL, and KuaiRand-Pure.
- `fairness/public_group_candidates.py`: current group-bearing candidate runner
  for the public fairness datasets.
- `fairness/public_auto_repro_run.py` and
  `fairness/verify_public_auto_repro.py`: frozen held public-auto reproduction
  and verification scripts with source, split, model, and output provenance.
- `docs/code_algorithm_audit_20261010.md`: the implementation-to-algorithm audit
  covering the group-price denominator wording fix and the raw-scale auxiliary
  proxy repair.

Supply local copies of the public datasets when running the scripts.

## Fairness outcome models

The current fairness runners train W-BCE on raw value and loss. Separate
squared-error linear regressions fit raw value and loss targets for complaint
construction. Training-only centering and scaling are inverted at prediction;
negative predicted means are projected to zero, and value proxies used in ratios
have a 1e-9 numerical floor. The auxiliary targets are not logged.

This replaces the earlier log1p-target auxiliary regressions. Main raw W-BCE
training was already correct. The separately labelled log-damped W-BCE
score-learning robustness comparison remains distinct.

The regenerated seven-source comparison at the `.95` validation capture floor
has median held-out closure lift +0.5010, source signs 7/0/0, and two-sided
source-cluster p=0.008106 over 149 eligible paired cells. The previous +0.6131
result used 180 cells and superseded log proxies; the eligibility change is
part of the old/new comparison. Fixed block-count and blocked-value parity
comparisons are mixed, so these numbers do not support dominance over every
parity baseline. See [the audit](docs/code_algorithm_audit_20261010.md).

Independent replay checked 120 splits, 360 saved models and 1,335,600 candidate
rows against raw inputs in the evidence bundle. Exact objection-mass reporting
is now complete for 2,484/2,484 value-unit selected policies; final paper
rendering/review remain unfinished.

## Inputs

- Australian MTPL: `ausprivauto0405.rda` from CASdatasets.
- Belgian MTPL: `beMTPL97.rda` from CASdatasets.
- Held public-auto reproduction: CASdatasets at commit
  `227fb56b8734bdb7c0327a41180e01d2ddaeaf26`, using `beMTPL97`,
  `freMPL1`--`freMPL9`, `norauto`, and `swmotorcycle`.
- KuaiRand-Pure: the extracted `data` directory containing
  `log_random_4_22_to_5_08_pure.csv`, `user_features_pure.csv`, and
  `video_features_basic_pure.csv`.

## Environment

Install the public dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

For the exact held public-auto reproduction, install the pinned runner
environment instead:

```bash
python -m pip install -r fairness/public_auto_requirements.txt
```

## Run

Score-learning runners live under `score/`.  Each script exposes its input and
output arguments through `--help`; generated CSVs should stay in a local output
directory and are not committed here.

Export CASdatasets inputs when needed:

```bash
python fairness/export_cas_rda_to_csv.py \
  --input <path/to/ausprivauto0405.rda> \
  --output <path/to/ausprivauto0405.csv>
```

Generate candidate-level logs:

```bash
python fairness/fairness_candidates.py \
  --dataset australian_mtpl \
  --input <path/to/ausprivauto0405.csv> \
  --dataset-scope all \
  --seeds 10 \
  --epochs 20 \
  --targets 0.1 \
  --output <output-dir>/australian_candidates.csv
```

Run the frozen held public-auto reproduction:

```bash
git clone https://github.com/dutangc/CASdatasets <path/to/CASdatasets>
git -C <path/to/CASdatasets> checkout 227fb56b8734bdb7c0327a41180e01d2ddaeaf26
python fairness/public_auto_repro_run.py --raw-repo <path/to/CASdatasets>
python fairness/verify_public_auto_repro.py --raw-repo <path/to/CASdatasets>
```

See `fairness/RUN_COMMANDS.md` for Belgian MTPL and KuaiRand examples.  Save the Git
commit, exact command, pinned requirements, raw input source URLs, raw input
hashes, generated CSV paths, and CSV checksums with any run.
