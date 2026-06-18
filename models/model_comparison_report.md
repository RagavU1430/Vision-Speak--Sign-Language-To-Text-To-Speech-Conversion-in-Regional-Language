# VisionSpeak Sign Language Model Benchmarking Report

This report benchmarks the original landmark-based MLP model against the direct image-based CNN classifiers.

## 1. Performance Comparison Table

| Metric | MLP Classifier (Landmark) | CNN / EfficientNet (Direct Image) |
|---|---|---|
| Model Architecture | MLP (Landmarks) | N/A |
| Model File Size | 2.51 MB | N/A |
| Mean Latency | 1.99 ms | N/A |
| Throughput | 501.61 FPS | N/A |
| Input Representation | 1D Array (99 features) | N/A |
| Real-World Robustness | Medium (sensitive to hand detection jitter) | N/A |


## 2. Analysis & Recommendations

> [!NOTE]
> **Landmark MLP**: Offers ultra-low latency and micro model size (<3MB), making it extremely CPU-friendly. However, it relies heavily on MediaPipe tracking precision; if hand lines are partially occluded, accuracy drops quickly.

> [!TIP]
> **Direct Image CNN**: Possesses superior translation invariance and shape recognition. While larger in disk footprint (~15-20MB for custom, ~40MB for EfficientNet) and higher in computation latency, it maintains high accuracy in complex backgrounds and lightings.
