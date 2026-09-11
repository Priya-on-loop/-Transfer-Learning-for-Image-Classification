#  Transfer Learning for Image Classification on CIFAR-10

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![GPU Training](https://img.shields.io/badge/GPU-Nvidia%20T4%20%2F%20A100-green?logo=nvidia)](https://developer.nvidia.com/)
[![Mixed Precision](https://img.shields.io/badge/Precision-Mixed%20FP16-blue)](https://www.tensorflow.org/guide/mixed_precision)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-performance image classification pipeline using **Transfer Learning with a ResNet-50 backbone** on the benchmark **CIFAR-10** dataset. This project implements production-grade deep learning practices including GPU-side data processing, mixed-precision arithmetic, and systematic regularization to achieve high generalization accuracy.

---

##  Project Overview
The objective is to classify 32x32 pixel RGB images across 10 distinct classes using a deep convolutional network pretrained on ImageNet. 

### Key Technical Challenges Solved:
1. **OOM RAM Crash Prevention**: Instead of upscaling images on the CPU, we built a custom pipeline that streams raw 32x32 images directly to the GPU, where resizing ($32 \to 224$), normalization, and augmentation are performed on-the-fly. This reduced host system RAM usage from **12 GB to < 1 GB**.
2. **Speed Optimization**: Integrated `mixed_float16` precision policy, cutting GPU memory utilization in half and increasing training throughput by **2.5x** on a Tesla T4.

---

##  Architecture Blueprint

The complete model architecture is structured as follows:

```text
Input (32x32x3 RGB)
   │
   ├── [GPU-Prepro] Random Horizontal Flip & Rotation (0.1)
   ├── [GPU-Prepro] Spatial Bilinear Resizing to (224x224x3)
   ├── [GPU-Prepro] Channel-wise Standardization (ImageNet Stats)
   │
   ├── [Backbone]   ResNet-50 (Pretrained on ImageNet, Frozen/Unfrozen)
   │
   └── [Custom Top] Global Average Pooling 
