# NJ_3_Aug — Run Summary

**Date:** 2025-10-26  
**Model:** ResNet-18  
**Epochs:** 5  
**Max Train Accuracy:** 81.10%  
**Console Log:** `runs/NJ_3_Aug_console.log`

---

## 🦴 Skipped Classes (Fewer than 20 Baseline Images)

| Taxon | Baseline Images |
|-------|-----------------|
| taxon-vertebra_(chondrichthyes) | 19 |
| taxon-ophiomorpha_nodosa | 17 |
| taxon-bone_(reptilia) | 3 |
| taxon-ischyrhiza_mira | 14 |

---

## 📦 Packaged Artifacts

The folder `Test Results/NJ_3_Aug/` contains:

- `NJ_3_Aug_console.log`
- `skipped_taxa.txt`
- `python_version.txt`
- `pip_freeze.txt`
- `git_state.txt` *(if available)*
- `README_NJ_3_Aug.txt`

> ⚠️ Note: No `hparams.json` or `summary.csv` were generated for this run.

---

## 🧠 Notes

- Run executed successfully using the patched `augment_images.py` (cross-platform via `os.sep`).
- GPU not detected — model trained on CPU (slower epochs).
- Dataset sourced from owner folders under `data/train` (with `DNU` moved out).
- All classes except the four listed above met the baseline threshold and were used in training.

---

**Prepared by:** *Mohamed Ernest Lebbie*  
**Environment:** macOS (Python 3.12, venv active)  
**Repository:** `capstone-fossil-gui`