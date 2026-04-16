## 🏗️ System Architecture

```text
[Capture Thread]
        ↓
[Preprocess Thread]
        ↓
[Inference Thread (TensorRT / GPU)]
        ↓
[Postprocess Thread (CPU)]
        ↓
[Display Thread]

```
Data flows through thread-safe queues with frame dropping strategy to ensure real-time performance.

# 🚀 High-performance Real-time Edge AI Inference System
(TensorRT + Multi-thread + Asynchronous Pipeline)

# Project Overview
This project implements a high-performance real-time object detection system using TensorRT in C++.

A multi-threaded pipeline is designed to decouple video capture, preprocessing, inference, and postprocessing. 
Cross-frame asynchronous execution is introduced to overlap CPU and GPU workloads, significantly improving system throughput.

The system achieves stable real-time performance (~30+ FPS) on a desktop GPU.

# System Architecture
The system follows a producer-consumer architecture with multiple threads:

Capture Thread → Preprocess Thread → Inference Thread → Postprocess Thread → Display Thread
                        ↓
                    Task Queue
                        ↓
              GPU (TensorRT Stream Execution)

# Pipeline Design
- Producer: continuously captures frames
- Preprocess: letterbox, normalization, CHW conversion
- Inference: TensorRT FP16 execution on GPU
- Postprocess: decode + NMS + rendering
- Consumer: real-time display

Data is passed through Task structures across threads.
Frame dropping is applied to ensure real-time performance.

# Performance optimization
Performance improvements through different stages:

| Stage                      | FPS       |
|---------------------------|----------|
| Single-thread baseline     | ~5 FPS    |
| Multi-thread pipeline      | ~10 FPS   |
| TensorRT FP16 inference    | ~20 FPS   |
| Cross-frame async (overlap)| ~30+ FPS  |

Key optimizations:
- Eliminated CPU-GPU synchronization bottleneck
- Implemented double buffering
- Introduced frame dropping strategy
- Avoided runtime memory allocation (cudaMalloc)

# Technical Highlights
- Multi-threaded real-time inference pipeline design
- Cross-frame asynchronous execution (CPU/GPU overlap)
- TensorRT FP16 acceleration
- Double buffering for efficient memory reuse
- Real-time system design with frame dropping strategy
- Performance bottleneck analysis (CPU-bound system)

# Technology Stack
- C++
- TensorRT
- CUDA
- OpenCV
- Multi-threading (std::thread, mutex, condition_variable)

## Demo

![demo](demo.gif)