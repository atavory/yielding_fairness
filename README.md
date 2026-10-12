# Yielding Fairness

Public repository: [atavory/yielding_fairness](https://github.com/atavory/yielding_fairness).

This repository contains public Python runners for the value-weighted blocking
experiments.  Raw datasets remain external. The frozen paper-result inputs and original
table/figure generators are available in
[the reproduction archive](artifacts/frozen_paper_evidence_20261011.zip); see
[the complete commands and artifact map](docs/paper_reproduction.md).

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

The current arXiv preparation uses the corrected normalized, enlarged price
bank in [the 20261012 evidence archive](artifacts/frozen_normalized_price_evidence_20261012.zip).
At the `.95` validation capture floor, the mean of seven source medians is
+0.2185 (two-sided source p=0.3071); across all ten recorded sources it is
+0.0863 (p=0.6007). These comparisons do **not** establish statistically
significant average group-aware superiority. An independent wider global grid gives ten-source mean +0.0202 (p=0.8990),
with the same conclusion. Historical stronger results used a truncated bank
with different effective price units.

The new archive retains every original physical policy, all recorded sources,
three measurement units, validation coverage traces and frozen LP weights.
Independent checks passed all 2,880 LP cells and 192 original scalar-policy
replays. It includes all frozen inputs needed to independently verify LP
optima and regenerate manuscript tables without raw datasets. Exact candidate
re-evaluation requires the external frozen candidate logs listed by checksum.

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

## Reproduce grouped Figure 3

The single-column grouped score chart uses the original twenty paired means
and 95% intervals. Extract `artifacts/bandha_score_grouped_figure_20261012.zip`
into a separate directory, install its `requirements.txt`, and run:

```bash
python3 viz/bandha_make_score_grouped_figure.py --base-dir .
```

The archive includes the generator, frozen summary CSVs, vector figure, and
input/output checksums. It regenerates the chart without retraining.
