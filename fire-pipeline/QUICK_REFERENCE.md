# Quick Reference: TensorBoard & Metrics Analysis

## TensorBoard - 3 Steps to Use

### Step 1: Train (TensorBoard enabled by default)
```bash
cd fire-pipeline
uv run python train.py --patches-dir ./patches
```

### Step 2: View in Browser
```bash
tensorboard --logdir=output/tensorboard
# Open: http://localhost:6006
```

### Step 3: Compare Runs (Optional)
```bash
# If you have multiple training runs:
tensorboard --logdir=output  # Parent directory with multiple runs
```

---

## Metrics Analysis - Quick Commands

### Single Model
```bash
python analyze_metrics.py
# or with custom paths:
python analyze_metrics.py --metrics output/metrics.csv --output-dir plots
```

### Compare 2 Models
```bash
python analyze_metrics.py \
    --metrics run1/metrics.csv \
    --metrics run2/metrics.csv \
    --model-names "Model 1" "Model 2"
```

### Compare 3+ Models
```bash
python analyze_metrics.py \
    --metrics model_a/metrics.csv \
    --metrics model_b/metrics.csv \
    --metrics model_c/metrics.csv \
    --model-names "A" "B" "C" \
    --compare --output-dir comparison_plots
```

---

## What Gets Logged

### TensorBoard (Real-Time)
```
train/loss, train/fire_iou, train/detection_f1
val/loss, val/fire_iou, val/fire_recall, val/detection_f1
lr (learning rate)
```

### Metrics CSV (Post-Analysis)
```
Same metrics saved to: output/metrics.csv
Can be analyzed anytime with analyze_metrics.py
```

---

## Generated Plots

### Single Model
```
fire_iou.png
fire_dice.png
mean_iou.png
detection_f1.png
detection_recall.png
detection_precision.png
loss.png
```

### Multi-Model Comparison
```
fire_iou_comparison.png
fire_dice_comparison.png
detection_f1_comparison.png
detection_recall_comparison.png
loss_comparison.png
metrics_comparison_grid.png    ← Most useful!
+ Console summary with best values
```

---

## Common Scenarios

### During Development
1. Train with `uv run python train.py ...`
2. Open TensorBoard in another terminal: `tensorboard --logdir=output/tensorboard`
3. Watch metrics update in real-time

### After Training
```bash
# Quick look at single run
python analyze_metrics.py

# Share results - compare with baseline
python analyze_metrics.py \
    --metrics baseline/metrics.csv \
    --metrics new_model/metrics.csv \
    --model-names "Baseline" "Improved" \
    --compare
```

### Before Publishing
```bash
python analyze_metrics.py \
    --metrics final_model/metrics.csv \
    --output-dir paper_figures
# Use PNG files in paper
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No TensorBoard? | `pip install tensorboard` |
| Port 6006 in use? | `tensorboard --logdir=... --port 6007` |
| Comparison plots empty? | Check that all CSV files have same metric columns |
| Metrics CSV not found? | Default is `output/metrics.csv`, use `--metrics` to specify |

---

## Key Arguments

### Training
```bash
# Disable TensorBoard if not wanted
--no-tensorboard
```

### Analysis
```bash
--metrics <PATH>           # Multiple times for comparison
--model-names <NAME>       # Multiple times for custom names
--output-dir <DIR>         # Where to save plots
--split <train|val>        # Which split to analyze
--compare                  # Force comparison mode
--single                   # Force single-model mode
```

---

## Tips

✅ **Best Practice**: Run TensorBoard during training, analyze metrics after
✅ **For Comparison**: Use meaningful model names with `--model-names`
✅ **For Papers**: Save comparison plots for side-by-side model evaluation
✅ **For Debugging**: Check TensorBoard for weird training behavior
✅ **For Sharing**: Use `metrics_comparison_grid.png` to show model differences

---

## Example: Complete Workflow

```bash
# Step 1: Train model A
uv run python train.py --patches-dir ./patches --output-dir output_a

# Step 2: Train model B (different architecture)
uv run python train.py --patches-dir ./patches \
    --encoder efficientnet-b0 \
    --output-dir output_b

# Step 3: Compare
python analyze_metrics.py \
    --metrics output_a/metrics.csv \
    --metrics output_b/metrics.csv \
    --model-names "ResNet34" "EfficientNet-B0" \
    --compare --output-dir final_comparison

# Step 4: View results
# - TensorBoard: tensorboard --logdir=output_a/tensorboard
# - Plots: Check final_comparison/metrics_comparison_grid.png
# - Summary: Check console output with best metrics
```

---

Last Updated: January 30, 2026
