# Paper Sync, 2026-10-10

## Current raw-outcome sync

- Data/evidence state: e039b9119162d2d60cc69968bd0a460df42fdb4e.
- Public raw-outcome runner baseline: 96835ee1fea17c5c2deee90f7d82d397b3922847.
- Paper raw-result/provenance sync: 79514dd5e3d0247cc56d6752ef2e2e1f6059cca6.
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

Current references:

- Paper Overleaf commit:
  `79514dd5e3d0247cc56d6752ef2e2e1f6059cca6`
- Data/evidence Overleaf commit:
  `e039b9119162d2d60cc69968bd0a460df42fdb4e`
- Public mirror runner baseline before this documentation refresh:
  `96835ee1fea17c5c2deee90f7d82d397b3922847`

The evidence bundle now closes the selected-policy mass accounting gate:
2,484/2,484 value-unit selected policies have exact repaired mass rows and
zero validation budget violations. Those internal mass artifacts are not
shipped in this public mirror.

Latest local compile proof on record:

  Paper commit `cd69ad8afaa25131ff70981af7a960b895e77ae4`,
  `main.pdf` SHA-256
  `db03bb4564dd18cad1d788e06cbe91129c7a8b17115df326a8b4442baa992d65`

That local Tectonic compile had 13 pages, references beginning on page 8, the
appendix beginning on page 9, and zero fatal, undefined-reference, or rerun
warnings. It predates the current paper sync. The remaining submission gate is
to compile the current Overleaf source and record matching PDF/source
provenance.
