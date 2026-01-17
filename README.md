# Mehfooz Musafir: Automated Helmet Detection System

**A Computer Vision pipeline for road safety compliance using YOLOv8.**
![Python](https://img.shields.io/badge/python-3.11-blue.svg) ![Framework](https://img.shields.io/badge/YOLO-v8-green)

## Overview

**Mehfooz Musafir** ("Safe Traveler") is an object detection system designed to identify motorcyclists and assess helmet compliance in real-time. The project leverages the **YOLOv8** (You Only Look Once) architecture, optimized for deployment on edge devices or cloud processing pipelines.

This repository contains the training source code, validation logic, and deployment scripts. The training workflow is architected to utilize **Google Colab (T4 GPU)** for compute while maintaining code versioning locally via **VS Code**.

## Model Performance
The model was trained on a custom-engineered dataset (merged from multiple sources) to address class imbalance. It achieves production-grade detection rates for the critical safety class ("No Helmet").

- Primary Success: The system excels at its main task, achieving an 81.1% mAP for detecting "No Helmet" violations on completely new data.

- Safety First: With a Recall of 0.78, the model catches nearly 8 out of 10 offenders, prioritizing the detection of violations over perfect bounding box precision.

- Generalization: The test results closely mirror the validation scores (0.83 vs 0.81), proving the model has not overfit and functions well in diverse, real-world conditions.

- Area for Improvement: Classes like "Half-Faced" (0.65 mAP) and "Full-Faced" (0.69 mAP) show moderate performance, likely due to visual overlaps with valid/invalid helmet types.

## Generalization Check
To ensure no overfitting occurred, we compared the Training/Validation results against the unseen Test set:

Validation Score (No Helmet): 0.83 mAP

Test Score (No Helmet): 0.81 mAP

Conclusion: The minimal drop (<2%) confirms the model generalizes exceptionally well to real-world scenarios.

## System Architecture

The project utilizes a hybrid cloud-local workflow to mitigate hardware limitations while ensuring data persistence.

* **Compute Engine:** Google Colab (Tesla T4 GPU).
* **Storage & Persistence:** Google Drive (Mount point for datasets and model checkpoints).
* **Versioning:** Git/GitHub (Source code control).
* **Framework:** Ultralytics YOLOv8 (PyTorch backend).

### I/O Optimization Strategy
To address network latency inherent in cloud-mounted storage, the training pipeline implements an automated **Ephemeral Storage Strategy**:
1.  **Ingestion:** Training scripts detect and pull compressed datasets from Google Drive.
2.  **Caching:** Data is extracted to the Colab VM's local high-speed SSD (`/content/dataset`).
3.  **Sanitization:** Scripts automatically normalize directory structures (flattening nested folders) to match YOLO configuration requirements.
4.  **Checkpointing:** Model weights (`best.pt`, `last.pt`) are synchronously saved back to Google Drive after every epoch to prevent data loss during runtime disconnects.

## Repository Structure

Note: Large assets (datasets, training logs, and binary weights) are excluded from version control via `.gitignore`.

```text
Mehfooz_Musafir/
├── src/                 # Source code for training and inference
│   ├── train.py         # Primary entry point for model training
│   └── validate.py      # Validation and metrics evaluation
├── .gitignore           # Configuration for excluded artifacts
└── README.md            # Project documentation