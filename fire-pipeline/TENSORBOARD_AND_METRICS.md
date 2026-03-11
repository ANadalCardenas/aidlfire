# TensorBoard and Metrics Analysis Guide

This guide explains the TensorBoard integration and enhanced metrics analysis features in the fire detection pipeline.

## TensorBoard Integration

### What's New

The training script now includes built-in TensorBoard logging for real-time visualization of training progress.

### Features

TensorBoard logs the following metrics:

**Training Metrics:**
- `train/loss` - Training loss
- `train/fire_iou` - Fire IoU on training set
- `train/detection_f1` - Detection F1 score on training set
- `train/ce_loss` - Cross-entropy loss component (if available)
- `train/dice_loss` - Dice loss component (if available)

**Validation Metrics:**
- `val/loss` - Validation loss
- `val/fire_iou` - Fire IoU on validation set
- `val/fire_recall` - Fire detection recall on validation set
- `val/detection_f1` - Detection F1 score on validation set

**Optimizer Metrics:**
- `lr` - Learning rate (useful for visualizing LR schedule)

### Usage

#### Enable TensorBoard (default)

TensorBoard is **enabled by default**. Just run training normally:

```bash
cd fire-pipeline
uv run python train.py --patches-dir ./patches --output-dir ./output
```

#### Disable TensorBoard if needed

```bash
uv run python train.py --patches-dir ./patches --no-tensorboard
```

#### View TensorBoard

While training or after it completes:

```bash
tensorboard --logdir=./output/tensorboard
```

Then open your browser to `http://localhost:6006`

#### Multi-run Comparison in TensorBoard

You can compare multiple training runs by pointing TensorBoard to the parent directory:

```bash
# If you have multiple training runs in separate folders
tensorboard --logdir=./output
```

This will show all TensorBoard logs from subdirectories, allowing side-by-side metric comparison.

---

## Enhanced Metrics Analysis

### What's New

The `analyze_metrics.py` script has been upgraded to support:
- **Single-model analysis** (original functionality)
- **Multi-model comparison** (new feature)
- **Grid plots** (multiple metrics in one figure)
- **Summary statistics** (best metrics per model)

### Single Model Analysis (Default)

Analyze a single training run:

```bash
cd fire-pipeline
python analyze_metrics.py --metrics output/metrics.csv --output-dir output/plots
```

This generates individual plots for:
- `fire_iou.png`
- `fire_dice.png`
- `mean_iou.png`
- `detection_f1.png`
- `detection_recall.png`
- `detection_precision.png`
- `loss.png`

### Multi-Model Comparison

Compare multiple models side-by-side. This is useful for comparing:
- Different architectures (U-Net vs U-Net++ vs DeepLabV3+)
- Different encoders (ResNet18 vs ResNet34 vs EfficientNet)
- Different training configurations
- Different loss functions

#### Example: Compare 3 models

```bash
python analyze_metrics.py \
    --metrics model1/output/metrics.csv \
    --metrics model2/output/metrics.csv \
    --metrics model3/output/metrics.csv \
    --model-names "ResNet34" "EfficientNet-B0" "ResNet50" \
    --output-dir comparison_plots \
    --compare
```

This generates:
- `fire_iou_comparison.png` - Side-by-side IoU curves
- `detection_f1_comparison.png` - Side-by-side F1 curves
- `loss_comparison.png` - Side-by-side loss curves
- `metrics_comparison_grid.png` - 2x2 grid with all key metrics
- Console output with summary statistics

#### Output Example

```
================================================================================
MODEL COMPARISON SUMMARY
================================================================================

ResNet34:
----------------------------------------
  fire_iou            : 0.7234 (epoch 42)
  fire_dice           : 0.8401 (epoch 42)
  detection_f1        : 0.9123 (epoch 38)
  detection_recall    : 0.8945 (epoch 38)
  loss                : 0.2341 (epoch 42)

EfficientNet-B0:
----------------------------------------
  fire_iou            : 0.7456 (epoch 35)
  fire_dice           : 0.8523 (epoch 35)
  detection_f1        : 0.9234 (epoch 33)
  detection_recall    : 0.9045 (epoch 33)
  loss                : 0.2189 (epoch 35)
```

### Script Options

```bash
python analyze_metrics.py --help
```

**Key Arguments:**

- `--metrics <PATH>` - Path to metrics.csv file. Can specify multiple times for comparison.
- `--model-names <NAME>` - Model names for legend. Can specify multiple times in same order as `--metrics`.
- `--output-dir <DIR>` - Directory to save plots (default: output/plots)
- `--split <SPLIT>` - Which split to analyze: 'train' or 'val' (default: val)
- `--compare` - Enable multi-model comparison mode (auto-enabled with 2+ metrics)
- `--single` - Force single-model analysis mode

### Workflow Examples

#### Scenario 1: Quick analysis of one training run

```bash
python analyze_metrics.py
```

This uses default paths: reads from `output/metrics.csv`, saves to `output/plots`

#### Scenario 2: Compare two loss functions

```bash
python analyze_metrics.py \
    --metrics output_ce/metrics.csv \
    --metrics output_focal/metrics.csv \
    --model-names "CrossEntropy" "Focal Loss" \
    --output-dir plots/loss_comparison
```

#### Scenario 3: Compare different architectures across multiple metrics

```bash
python analyze_metrics.py \
    --metrics runs/unet_resnet34/metrics.csv \
    --metrics runs/unet_efficientnet/metrics.csv \
    --metrics runs/unetplusplus_resnet34/metrics.csv \
    --model-names "U-Net (R34)" "U-Net (ENet)" "U-Net++ (R34)" \
    --output-dir comparison_plots \
    --split val \
    --compare
```

---

## Combining TensorBoard + analyze_metrics.py

### Recommended Workflow

1. **During Training**: Monitor metrics with TensorBoard
   ```bash
   tensorboard --logdir=output/tensorboard &
   uv run python train.py --patches-dir ./patches
   ```

2. **After Training**: Generate publication-ready plots
   ```bash
   python analyze_metrics.py --metrics output/metrics.csv --output-dir output/plots
   ```

3. **Comparing Models**: Generate comparison plots
   ```bash
   python analyze_metrics.py \
       --metrics output1/metrics.csv \
       --metrics output2/metrics.csv \
       --compare --output-dir comparison_plots
   ```

### Why Both Tools?

- **TensorBoard**: Interactive, real-time, great for debugging during training
- **analyze_metrics.py**: Static plots, publication-ready, easy to share, supports model comparison

---

## Troubleshooting

### "TensorBoard not found" error

Install TensorBoard:
```bash
pip install tensorboard
```

Or with uv:
```bash
uv pip install tensorboard
```

### Metrics CSV not found

By default, the script looks for `output/metrics.csv`. If your metrics are elsewhere, specify the path:

```bash
python analyze_metrics.py --metrics /path/to/metrics.csv
```

### Comparison plots look empty

Make sure:
1. All metrics files exist and are readable
2. All models have the same metric columns
3. You specified `--compare` flag (or have 2+ metrics files)
4. Metrics files are valid CSV with "epoch" and "split" columns

### TensorBoard shows no data

Check:
1. Training is saving to `output/tensorboard` (default)
2. `--no-tensorboard` flag was not used
3. TensorBoard logdir points to the correct directory

---

## File Structure

After training with TensorBoard enabled:

```
output/
├── metrics.csv          # CSV metrics (used by analyze_metrics.py)
├── config.json          # Training configuration
├── tensorboard/         # TensorBoard event files
│   └── events.out.tfevents.*
├── checkpoints/
│   ├── best_model.pt
│   ├── final_model.pt
│   └── checkpoint_epoch_*.pt
└── plots/               # Generated by analyze_metrics.py
    ├── fire_iou.png
    ├── fire_dice.png
    ├── detection_f1.png
    └── ...
```

---

## Tips for Best Results

### For TensorBoard

- Use consistent learning rates across runs for easier comparison
- Train for similar numbers of epochs
- Use the same hardware/batch size to avoid noise

### For analyze_metrics.py

- Use meaningful model names in `--model-names`
- Save metrics to timestamped directories: `output_2024-01-30_001/`
- Generate comparison plots at the end of all training

### For Publication Figures

1. Generate plots with `analyze_metrics.py`
2. Adjust font sizes/colors in the script if needed
3. Plots are saved at 150 DPI (good for print)
4. Use comparison plots to highlight differences between models
