# Resource-Efficient ML Inference API

A lightweight ML deployment system for compressed image classification.

## Features
- MobileNetV2 fine-tuned on CIFAR-10
- ONNX export for deployment
- INT8 quantization for size reduction

## Project Structure
```
MLH_PROJECT/
├── training/          # Training and export scripts
├── models/            # ONNX models
├── app/              # API (coming soon)
└── tests/            # Tests (coming soon)
```

## Setup
```bash
pip install -r requirements.txt
cd training
python train.py
python export_onnx.py
```

## MLH Fellowship Application
This project demonstrates production ML deployment for the MLH Fellowship.