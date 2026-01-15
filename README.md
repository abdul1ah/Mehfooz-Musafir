# Mehfooz Musafir: Automated Helmet Detection System

**A Computer Vision pipeline for road safety compliance using YOLOv8.**
![Python](https://img.shields.io/badge/python-3.11-blue.svg) ![Framework](https://img.shields.io/badge/YOLO-v8-green)

## Overview

**Mehfooz Musafir** ("Safe Traveler") is an object detection system designed to identify motorcyclists and assess helmet compliance in real-time. The project leverages the **YOLOv8** (You Only Look Once) architecture, optimized for deployment on edge devices or cloud processing pipelines.

This repository contains the training source code, validation logic, and deployment scripts. The training workflow is architected to utilize **Google Colab (T4 GPU)** for compute while maintaining code versioning locally via **VS Code**.

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