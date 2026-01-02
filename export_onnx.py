import torch
import torch.nn as nn
from torchvision import models
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# For model verification
import onnxruntime as ort
import numpy as np

# configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10

CHECKPOINT_PATH = "./checkpoints/mobilenet_cifar10_fp32.pth"
ONNX_FP32_PATH = "./models/mobilenet_cifar10_fp32.onnx"
ONNX_INT8_PATH = "./models/mobilenet_cifar10_int8.onnx"

os.makedirs("./models", exist_ok=True)