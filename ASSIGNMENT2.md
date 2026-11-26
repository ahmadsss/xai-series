# WQF7009 Assignment 2: XAI on brain MRI (LRP + Grad-CAM)

Fork of [deepfindr/xai-series](https://github.com/deepfindr/xai-series) used for the assignment.

## Contents added for the assignment
| Path | Purpose |
|---|---|
| `data/brain_mri/{training,testing}/` | Brain MRI images from [SartajBhuvaji/Brain-Tumor-Classification-DataSet](https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet), commit `3ea9f4e`, top-level folders renamed to lowercase |
| `05_lrp.py` | Starter script (training). Step 3 of `apply_lrp_on_vgg16` was changed in the Colab run to build the one-hot output in torch (NumPy 2.x `TypeError`) |
| `06_xai_brain_mri.py` | Loads the trained checkpoint, full test-set evaluation, LRP, Grad-CAM, case figures, deletion test |
| `07_duplicate_check.py` | Readability and exact-duplicate check; writes `leaked_test_files.json` |
| `results.txt`, `figures/` | Outputs reported in the assignment report |

## Running (Google Colab, GPU runtime)
```python
!git clone https://github.com/ahmadsss/xai-series.git
%cd xai-series
# 1) Train with the training cells of 05_lrp.py (about 10 min), then:
#    torch.save(model.state_dict(), "vgg16_brain_mri_baseline.pth")
#    (or copy an existing checkpoint into this folder)
# 2) Evaluate and explain:
%run 06_xai_brain_mri.py
# 3) Data check (no GPU needed):
%run 07_duplicate_check.py
```
Environment used: torch 2.14.1+cu130, torchvision 0.29.1+cu130, numpy 2.4.6.
The checkpoint (513 MB) is not in the repository because of its size.
