#!/bin/bash
# Monitor training progress for: resnet50+unet, YOLO, Scratch

echo "================================"
echo "Training Status Monitor"
echo "================================"
echo ""

TIMESTAMP="20260208_172826"
OUTPUT_DIR="output/training_run_$TIMESTAMP"

echo "Output Directory: $OUTPUT_DIR"
echo ""

# Check segmentation model
if [ -d "$OUTPUT_DIR/encoder_resnet50-architecture_unet" ]; then
    echo "✅ Segmentation Model (resnet50 + unet):"
    CKPT=$(find "$OUTPUT_DIR/encoder_resnet50-architecture_unet" -name "*.pt" -type f 2>/dev/null | wc -l)
    echo "   Checkpoints found: $CKPT"
    if [ -f "$OUTPUT_DIR/encoder_resnet50-architecture_unet/metrics.json" ]; then
        echo "   Metrics available: YES"
    fi
else
    echo "⏳ Segmentation Model: Training in progress..."
fi

echo ""

# Check YOLO baseline
if [ -d "$OUTPUT_DIR/yolo_seg" ]; then
    echo "✅ YOLO Baseline:"
    CKPT=$(find "$OUTPUT_DIR/yolo_seg" -name "*.pt" -type f 2>/dev/null | wc -l)
    echo "   Checkpoints found: $CKPT"
else
    echo "⏳ YOLO Baseline: Waiting to start..."
fi

echo ""

# Check Scratch model
if [ -d "$OUTPUT_DIR/scratch_model" ]; then
    echo "✅ Scratch Model:"
    CKPT=$(find "$OUTPUT_DIR/scratch_model" -name "*.pt" -type f 2>/dev/null | wc -l)
    echo "   Checkpoints found: $CKPT"
else
    echo "⏳ Scratch Model: Waiting to start..."
fi

echo ""
echo "================================"
echo "📁 Full output in: $OUTPUT_DIR"
echo "================================"
