# EO4WildFires sample images

Sample figures for the EO4WildFires dataset docs are **generated** by running:

```bash
# From repo root; requires datasets==3.6.0 (incompatible with 4.x)
pip install "datasets==3.6.0" matplotlib numpy
python _docs_/scripts/export_eo4wildfires_samples.py
```

This downloads the validation split from [HuggingFace](https://huggingface.co/datasets/AUA-Informatics-Lab/eo4wildfires) and saves:

- `sample_s2_rgb.png` — Sentinel-2 true-color (first validation sample)
- `sample_burned_mask.png` — EFFIS burned area mask (first validation sample)
- `sample_s2_and_mask.png` — Side-by-side RGB + mask
- `sample_s2_rgb_0.png`, `sample_burned_mask_0.png`, etc. — Additional samples

If these files are missing, the dataset docs still render; only the image placeholders will be broken until you run the script.
