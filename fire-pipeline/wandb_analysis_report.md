# W&B Training Analysis Report

Analysis of experimental runs for postgrad project defense.

---

## 1. Run Overview

| Run Type | Count | Description |
|----------|-------|-------------|
| cnn_scratch | 11 | CNN from scratch (ScratchFireModel) |
| legacy_smp | 17 | Legacy SMP (CEMS-only, pretrained encoder) |
| unet_scratch | 6 | U-Net from scratch (no pretrained encoder) |
| v3_binary | 8 | Phase 1: Combined binary (CEMS DEL + Sen2Fire) |
| v3_severity | 14 | Phase 2: Severity fine-tune on CEMS GRA |
| yolo | 3 | YOLOv8-Seg baseline (RGB, detection-style) |

**Total runs:** 59 | **Finished:** 46

## 2. V3 Phase 1 (Combined Binary) — Best Models

Phase 1 trains binary fire detection on CEMS DEL + Sen2Fire.

| Rank | Architecture | Fire IoU | Det F1 | Fire Prec | Fire Rec | Run |
|------|--------------|----------|--------|-----------|----------|-----|
| 1 | resnet50_unetplusplus | 0.7791 | 0.8455 | 0.0000 | 0.9286 | [v3_combined_binary_resnet50_unetpp](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/7ql5q0zk) |
| 2 | resnet50_deeplabv3plus | 0.7724 | 0.8587 | 0.0000 | 0.9181 | [v3_combined_binary_resnet50_deeplabv3plu](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/z0q19ck5) |
| 3 | resnet18_unet | 0.7654 | 0.8351 | 0.0000 | 0.9218 | [v3_combined_binary_resnet18_unet](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/embjzx6b) |
| 4 | resnet34_unet | 0.7650 | 0.8524 | 0.0000 | 0.9286 | [v3_combined_binary_resnet34_unet](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/wf7soxxj) |
| 5 | efficientnet-b2_unetplusplus | 0.7612 | 0.8711 | 0.0000 | 0.9356 | [v3_combined_binary_efficientnet-b2_unetp](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/3xeahgrh) |
| 6 | mobilenet_v2_unet | 0.7572 | 0.8612 | 0.0000 | 0.9141 | [v3_combined_binary_mobilenet_unet](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/w2klain9) |
| 7 | efficientnet-b1_unetplusplus | 0.6736 | 0.8223 | 0.0000 | 0.9492 | [v3_combined_binary_efficientnet_unetpp](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/q8ibpk5p) |

**Top 3 Phase 1:**
- **#1** resnet50_unetplusplus: fire_iou=0.7791
- **#2** resnet50_deeplabv3plus: fire_iou=0.7724
- **#3** resnet18_unet: fire_iou=0.7654

## 3. V3 Phase 2 (Severity) — Best Models

Phase 2 fine-tunes severity head on CEMS GRA (5 classes).

| Rank | Architecture | Mean IoU | Fire Dice | Val Acc | Val AUC | Run |
|------|--------------|----------|-----------|--------|---------|-----|
| 1 | resnet50_unetplusplus | 0.3444 | 0.0000 | 0.0000 | 0.0000 | [v3_finetune_severity_resnet50_unetp](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/sb5t3qp4) |
| 2 | efficientnet-b2_unetplusplus | 0.3388 | 0.0000 | 0.0000 | 0.0000 | [v3_finetune_severity_efficientnet-b](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/nl5x5fj2) |
| 3 | efficientnet-b2_unetplusplus | 0.3364 | 0.0000 | 0.0000 | 0.0000 | [v3_finetune_severity_efficientnet-b](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/5ryow5t7) |
| 4 | resnet34_unet | 0.3352 | 0.0000 | 0.0000 | 0.0000 | [v3_finetune_severity_resnet34_unet](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/87r90end) |
| 5 | resnet34_unet | 0.3346 | 0.0000 | 0.0000 | 0.0000 | [v3_finetune_severity_resnet34_unet](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/kraosdtl) |
| 6 | efficientnet-b2_unetplusplus | 0.3341 | 0.0000 | 0.0000 | 0.0000 | [v3_finetune_severity_efficientnet-b](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/wjm2gvj7) |
| 7 | resnet50_deeplabv3plus | 0.3329 | 0.0000 | 0.0000 | 0.0000 | [v3_finetune_severity_resnet50_deepl](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/zr80otqj) |
| 8 | mobilenet_unet | 0.3317 | 0.0000 | 0.0000 | 0.0000 | [v3_finetune_severity_mobilenet_unet](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/pqvle9f1) |
| 9 | resnet18_unet | 0.3240 | 0.0000 | 0.0000 | 0.0000 | [v3_finetune_severity_resnet18_unet](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/slukfegt) |
| 10 | efficientnet_unetplusplus | 0.3167 | 0.0000 | 0.0000 | 0.0000 | [v3_finetune_severity_efficientnet_u](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/xihz71ht) |

**Top 3 Phase 2 (Severity):**
- **#1** resnet50_unetplusplus: mean_iou=0.3444, fire_dice=0.0000
- **#2** efficientnet-b2_unetplusplus: mean_iou=0.3388, fire_dice=0.0000
- **#3** efficientnet-b2_unetplusplus: mean_iou=0.3364, fire_dice=0.0000

## 4. Scratch Models (No Pretrained Encoder)

| Rank | Model | Fire Dice | Mean IoU | Val F1 | Run |
|------|-------|-----------|----------|-------|-----|
| 1 | unet_scratch | 0.8993 | 0.8909 | 0.0000 | [unet_scratch_e15_bs16_lr1e-3_wd1e-3](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/7lc0krfi) |
| 2 | unet_scratch | 0.8931 | 0.8854 | 0.0000 | [unet_scratch_e15_bs16_lr3e-4_wd1e-3](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/es1tvafz) |
| 3 | unet_scratch | 0.8384 | 0.8400 | 0.0000 | [unet-scratch+s2f-tune-top3-lr1.2e-04-wd2](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/s89w94t8) |
| 4 | unet_scratch | 0.8356 | 0.8366 | 0.0000 | [unet-scratch+s2f-tune-top1-lr1.8e-04-wd9](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/b0uf46zb) |
| 5 | unet_scratch | 0.7858 | 0.7906 | 0.0000 | [unet-scratch+s2f-tune-top2-lr2.8e-04-wd3](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/rmxkcwtd) |
| 6 | cnn_scratch | 0.0000 | 0.0000 | 0.6593 | [cnn_scratch_e15_bs16_lr3e-4_wd1e-3](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/bgun9efn) |
| 7 | cnn_scratch | 0.0000 | 0.0000 | 0.6527 | [cnn_scratch_e15_bs16_lr1e-3_wd1e-3](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/edqzk970) |
| 8 | cnn_scratch | 0.0000 | 0.0000 | 0.6077 | [scratch-tune-top2-lr1.2e-04-wd3.7e-06-do](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/wr692dpd) |

## 5. YOLO Baseline

YOLOv8-Seg (RGB detection-style). Different metric (mAP50-95).

| Rank | mAP50-95 | F1 | Precision | Recall | Run |
|------|----------|-----|-----------|--------|-----|
| 1 | 0.4397 | 0.6098 | 0.7130 | 0.5328 | [yolo+s2f-tune-top1-lr7.8e-03-wd6.7e-04-b](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/upbdth1b) |
| 2 | 0.4391 | 0.6079 | 0.7078 | 0.5328 | [yolo+s2f-tune-top2-lr2.9e-03-wd9.8e-04-b](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/b4y7pfte) |
| 3 | 0.4355 | 0.6021 | 0.7240 | 0.5153 | [yolo+s2f-tune-top3-lr9.5e-04-wd3.2e-04-b](https://wandb.ai/adrian-corvin-salceanu-upc-barcelona/fire-detection/runs/izbp833y) |

## 6. V3 Assumptions vs Experimental Data

From `docs/V3_PIPELINE_ARCHITECTURES.md`:

| Assumption | Expected | Data Support? |
|------------|----------|---------------|

### Phase 1 (Binary)

- **Default (resnet34 + unet):** Solid baseline, moderate size.
- **Higher accuracy (resnet50 + unetpp):** Better for fine-grained.
- **Best quality (efficientnet-b2 + unetpp, resnet50 + deeplabv3plus):** Best for severity.

**Actual Phase 1 ranking (by fire_iou / fire_dice):**
- #1 resnet50_unetplusplus
- #2 resnet50_deeplabv3plus
- #3 resnet18_unet
- #4 resnet34_unet
- #5 efficientnet-b2_unetplusplus

**Conclusion:**
- ✅ **resnet50 + unetpp** is among top performers — supports 'Higher accuracy' assumption.
- ✅ **resnet34 + unet** remains a solid baseline.

### Phase 2 (Severity)

**Actual Phase 2 ranking (by mean_iou):**
- #1 resnet50_unetplusplus
- #2 efficientnet-b2_unetplusplus
- #3 efficientnet-b2_unetplusplus
- #4 resnet34_unet
- #5 resnet34_unet

- ✅ **resnet50 + unetpp** is best for severity — supports assumption.

## 7. Inference Recommendation for App

**Recommended model for deployment:**

| Criterion | Recommendation | Rationale |
|-----------|----------------|-----------|
| **Best accuracy** | **resnet50_unetplusplus** | Top Phase 1 fire_iou and Phase 2 mean_iou. |
| **Balanced (accuracy + speed)** | **resnet34_unet** | Solid baseline, faster than ResNet50. |
| **Edge / lightweight** | **mobilenet_v2_unet** | Smallest footprint; acceptable accuracy. |

**Recommended for app:** Use **resnet50_unetplusplus** if compute allows; otherwise **resnet34_unet** for a good balance of speed and accuracy.

**Why not YOLO?** YOLO uses RGB-only; lower mAP50-95 (~0.44) than SMP segmentation (fire_iou ~0.78). SMP uses 8-channel (7 bands + NDVI) for better spectral discrimination.

**Why not scratch?** Scratch models (UNetScratch) can reach high fire_dice on CEMS-only, but CEMS+Sen2Fire combined training with pretrained SMP yields better generalization; pretrained encoders provide strong spectral feature extraction.

## 8. Additional Slices for Report

| Slice | Use in Report |
|-------|----------------|
| **By dataset** | CEMS-only vs CEMS+Sen2Fire — show benefit of combined training |
| **By input channels** | 7 vs 8 (NDVI) — validate vegetation index importance |
| **By training time** | Trade-off: ResNet50 vs MobileNet inference speed |
| **Precision vs Recall** | Fire detection: recall may matter more (don't miss fires) |
| **Failed/crashed runs** | Reproducibility, robustness of training setup |

## 9. Summary Table (Best per Architecture)

| Architecture | Phase 1 Fire IoU | Phase 2 Mean IoU | Phase 2 Fire Dice |
|--------------|-------------------|-------------------|------------------|
| efficientnet-b1_unetplusplus | 0.6736 | - | - |
| efficientnet-b2_unetplusplus | 0.7612 | 0.3388 | - |
| efficientnet_unetplusplus | - | 0.3167 | - |
| mobilenet_unet | - | 0.3317 | - |
| mobilenet_v2_unet | 0.7572 | - | - |
| resnet18_unet | 0.7654 | 0.3240 | - |
| resnet34_unet | 0.7650 | 0.3352 | - |
| resnet50_deeplabv3plus | 0.7724 | 0.3329 | - |
| resnet50_unetplusplus | 0.7791 | 0.3444 | - |
