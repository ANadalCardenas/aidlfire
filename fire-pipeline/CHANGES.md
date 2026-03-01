# Changelog / recent changes

## Vegetation (NDVI) extension

**Summary:** The pipeline now adds an **8th input channel (NDVI)** by default to improve separation of burn scars from other low-NIR surfaces (water, shadow, soil).

### Code and config changes

| Component | Change |
|-----------|--------|
| **constants.py** | `NUM_INPUT_CHANNELS = 8`, `NUM_SPECTRAL_BANDS = 7`, `RED_INDEX_7`, `NIR_INDEX_7`, `INPUT_CHANNEL_NAMES`, `BAND_DESCRIPTIONS` (+ NDVI) |
| **patch_generator.py** | `PatchConfig.include_ndvi` (default `True`); `_load_image()` computes NDVI from Red/NIR and appends as 8th channel |
| **run_pipeline.py** | `--no-ndvi` flag to disable NDVI (output 7 channels); prints channel count in summary |
| **train.py** | Uses `NUM_INPUT_CHANNELS` for model; saves `in_channels` in checkpoint config |
| **inference.py** | Reads `in_channels` from checkpoint (default 7 for old checkpoints); `preprocess_image()` adds NDVI when model expects 8 and input has 7 |

### Options reference

| Where | Option | Effect |
|-------|--------|--------|
| **run_pipeline.py** | (default) | 8 channels (7 bands + NDVI) |
| **run_pipeline.py** | `--no-ndvi` | 7 channels only |
| **PatchConfig** | `include_ndvi=True` | 8 channels (default) |
| **PatchConfig** | `include_ndvi=False` | 7 channels |
| **Checkpoint** | `config["in_channels"]` | 8 for new runs; 7 if missing (backward compat) |

### Backward compatibility

- **Old 7-channel checkpoints**: Load normally; inference uses `in_channels=7` if not in config.
- **New 8-channel runs**: Regenerate patches (no `--no-ndvi`), train as usual; checkpoints get `in_channels: 8`.
- **Mixed**: You can generate 7-channel patches with `--no-ndvi` and train; the code supports both 7 and 8 channels.

### Documentation updated

- **README.md**: Input channels and vegetation (NDVI) section, run_pipeline options, checkpoint config table, Stage 2/4 and inference notes.
- **PATCHES.md**: “The 8th channel (NDVI)” section, 7 or 8 channels in file structure and examples, pipeline options and steps.
- **CLAUDE.md**: Band table + NDVI, patch shape 7 or 8, generate-patches note, dataset shape.
- **AGENTS.md**: Input data (NDVI, constants), patches (7 or 8 channels), generate-patches task, important numbers, warnings.
- **docs/SEN2FIRE_INTEGRATION_PLAN.md**: Note that CEMS uses 8 channels by default; Sen2Fire integration should use same 8-channel spec.
