# Paper Sync, 2026-10-10

## Current raw-outcome sync

- Data regenerated state: 22108073145bace50770c5d113887fe9c2bc824d.
- Public raw-outcome runner: 96835ee1fea17c5c2deee90f7d82d397b3922847.
- Paper raw-result sync: 4c96b1c3f49c11826afeded6c060b7a9b0510948.
- Paper sign-flip caption correction: 3786444 (seven sources: minimum 1/128).

The headline and comparator tables were checked byte for byte against the
regenerated evidence artifacts, and recorded sync hashes matched. Final
Overleaf compilation and the full review harness are pending for this source.
The current distribution is the named-author arXiv version, with no rendered
country. The compile proof below applies to its recorded older commit only.

## Historical denominator-wording repair

The code-to-paper discrepancy audit found one paper wording error: the
group-aware complaint price is normalized by validation group-denominator value
masses, not validation complaint rates.

Resolved references:

- Paper Overleaf commit:
  `cd69ad8afaa25131ff70981af7a960b895e77ae4`
- Data/evidence Overleaf commit:
  `05212d1558ae8eff930511b9b78a2f8939075876`
- Local compile proof for the synced paper:
  `main.pdf` SHA-256
  `db03bb4564dd18cad1d788e06cbe91129c7a8b17115df326a8b4442baa992d65`

The local Tectonic compile had 13 pages, references beginning on page 8, the
appendix beginning on page 9, and zero fatal, undefined-reference, or rerun
warnings. That compile check is a source/PDF provenance gate; it is not an
anonymous-submission or page-limit gate.
