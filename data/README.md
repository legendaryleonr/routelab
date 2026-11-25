# Data

Datasets are **not** stored in this repository. `routelab data download` fetches
them into `data/raw/` (ignored by Git) from a fixed commit and checks every file
against the SHA-256 hash recorded in `src/routelab/datasets.py`. If a hash does
not match, the download fails and nothing is written.

## BANKING77

| | |
|---|---|
| Publisher | PolyAI |
| Source | <https://github.com/PolyAI-LDN/task-specific-datasets> |
| Commit | `57ec275d8078af65b7731c2a98be812d844a6d6b` |
| License | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| Paper | Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, NLP for ConvAI Workshop, ACL 2020. <https://arxiv.org/abs/2003.04807> |

| File | Rows | SHA-256 |
|---|---:|---|
| `banking_data/train.csv` | 10,003 | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| `banking_data/test.csv` | 3,080 | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |
| `banking_data/categories.json` | 77 labels | `53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63` |

RouteLab does not modify or redistribute these files. Model artifacts trained
on them are derived from BANKING77 and should carry the same attribution.

## Removing downloaded data

Delete the `data/raw/` folder. Nothing else depends on it after a model has been
trained.
