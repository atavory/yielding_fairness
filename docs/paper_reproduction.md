# Paper table and figure reproduction

The public repository supplies `artifacts/frozen_paper_evidence_20261011.zip`.
It contains the frozen aggregate inputs and the original generators for every
table and figure in the manuscript. It excludes individual raw dataset rows.
Its `MANIFEST.json` lists input hashes, the data revision, and expected output
hashes. Historical script names retain “Figure 3”; their common-frontier output
is manuscript Figure 4.

## Environment and commands

From a clone of https://github.com/atavory/yielding_fairness:

```bash
mkdir -p paper_reproduction
unzip artifacts/frozen_paper_evidence_20261011.zip -d paper_reproduction
python3 -m venv .paper-venv
. .paper-venv/bin/activate
python -m pip install scipy==1.18.0
cd paper_reproduction
mkdir -p results/tables results/figures
python scripts/bandha_generate_score_protocol_table.py
python scripts/bandha_generate_group_headline_table.py
python scripts/bandha_generate_matched_lp_table.py
python scripts/bandha_make_figure3_endpoint_table.py
python scripts/bandha_generate_fixed_parity_table.py
python viz/make_wsdm_objection_figures.py --series kdd
python viz/make_kdd_weighting_win_figures.py
python viz/bandha_make_figure3_common_frontier.py
```

Use Python 3.12 or newer and Git. All generators except the paired Student t
interval calculation in the score figure use the Python standard library.
Generate vector figures from their source; do not edit the generated PDFs.

| Manuscript artifact | Generator | Principal frozen input |
|---|---|---|
| Table 1: score samples | `scripts/bandha_generate_score_protocol_table.py` | `bandha_score_fixed_summary.csv` |
| Table 2: group-awareness floors | `scripts/bandha_generate_group_headline_table.py` | `bandha_combined_plus_euhealth_group_awareness_summary.csv`, `bandha_cluster_plus_euhealth_group_awareness_inference.csv` |
| Table 3: LP comparison | `scripts/bandha_generate_matched_lp_table.py` | `bandha_matched_account_frontier_lp_detail.csv`, independent LP verification |
| Table 4: account endpoints | `scripts/bandha_make_figure3_endpoint_table.py` | `bandha_figure3_policy_bank.csv`, `bandha_figure3_policy_summary.csv` |
| Appendix parity table | `scripts/bandha_generate_fixed_parity_table.py` | `bandha_fixed_parity_trichotomy_{detail,summary,eligibility}.csv` |
| Figures 1, 2, 5: illustrations | `viz/make_wsdm_objection_figures.py` | deterministic `build_data()` account construction |
| Figure 3: weighting comparisons | `viz/make_kdd_weighting_win_figures.py` | ten original split-level score CSVs |
| Figure 4: account frontier | `viz/bandha_make_figure3_common_frontier.py` | `bandha_figure3_policy_summary.csv`, `bandha_figure3_method_points.csv` |

Input CSVs reside in `results/csvs/`. The commands generate TeX tables under
`results/` and `results/tables/`, and figures under `results/figures/`.
Compare generated content against `expected_output_sha256` in `MANIFEST.json`.
Generated provenance includes the enclosing Git revision, so its revision
field changes when reproducing from the public repository. Scientific table
and figure contents must match the frozen expected outputs.

## Training and policy-bank reproduction

The public repository's `README.md`, `fairness/RUN_COMMANDS.md`, and pinned
fairness requirements describe candidate generation from locally supplied
public datasets. The frozen bundle above reproduces reported tables and
figures from fitted results. Full retraining additionally requires the source
datasets and their cohort/split settings; the archive does not grant access
to restricted credit or mortgage downloads. The full frozen Figure 4 candidate
bank is larger than this aggregate bundle; its data-project provenance retains
the portable candidate transports, hydration hashes, and zero-price replay.
