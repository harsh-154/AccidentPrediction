# Accident Prediction using Video Classification

A deep learning project for predicting accidents in video sequences using multiple state-of-the-art models including C3D, ResNet, and MobileNet.
Links of Dataset and Weights https://drive.google.com/drive/folders/1keCi4dgafXaM-Se5rWc_K3_dLlMWqKMa?usp=sharing
## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements video-based accident prediction using transfer learning with multiple deep learning architectures. The models are trained to classify video segments as either "Accident" or "Normal" using a custom dataset.

## 📊 Dataset

### Dataset Links

<!-- Add your dataset links here -->
- **Primary Dataset**: [Add dataset link here]
- **Alternative Datasets**: 
  - [Dataset source 1]
  - [Dataset source 2]

### Dataset Structure

The dataset should be organized in the following structure:

```
NiAD_Large_Videos/
├── Training/
│   ├── Accident/
│   │   └── [video files]
│   └── Normal/
│       └── [video files]
├── Validation/
│   ├── Accident/
│   │   └── [video files]
│   └── Normal/
│       └── [video files]
└── Testing/
    ├── Accident/
    │   └── [video files]
    └── Normal/
        └── [video files]
```

### Dataset Preparation

The project includes several scripts for video preprocessing:
- `segment_video_30frames.py` - Segments videos into 30-frame clips
- `segment_all_videos.py` - Batch processing for video segmentation
- `augment_accident_segments.py` - Data augmentation for accident segments
- `suffix_labels.py` - Label management utilities

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/harsh-154/AccidentPrediction.git
cd AccidentPrediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pretrained weights (if needed):
   - C3D pretrained weights: [Add link]
   - MobileNet pretrained weights: [Add link]
   - ResNet pretrained weights: [Add link]

## 📁 Project Structure

```
accident_prediction/
├── models/                  # Model architectures
│   ├── c3d.py              # C3D model implementation
│   ├── mobilenetv2.py      # MobileNetV2 model
│   └── resnet.py           # ResNet model
├── c3d70/                  # C3D model training/testing
│   ├── trainc3d.py        # Training script
│   ├── testc3dd.py        # Testing script
│   └── [results]           # Training outputs
├── mobilenet73/            # MobileNet model training/testing
│   ├── trainn73.py        # Training script
│   ├── test73.py          # Testing script
│   └── [results]           # Training outputs
├── resnet75/               # ResNet model training/testing
│   ├── train75.py         # Training script
│   ├── test.75.py         # Testing script
│   └── [results]           # Training outputs
├── weights/                # Pretrained model weights
├── NiAD_Large_Videos/      # Dataset directory (not in repo)
└── requirements.txt        # Python dependencies
```

## 🤖 Models

### 1. C3D (3D Convolutional Networks)
- **Architecture**: 3D CNN for spatiotemporal feature extraction
- **Pretrained**: C3D pretrained on Sports-1M
- **Configuration**: 16 frames per clip
- **Location**: `c3d70/`

### 2. MobileNetV2
- **Architecture**: Lightweight CNN optimized for mobile devices
- **Pretrained**: Kinetics-400 pretrained weights
- **Configuration**: Optimized for efficiency
- **Location**: `mobilenet73/`

### 3. ResNet
- **Architecture**: Residual Network with transfer learning
- **Pretrained**: ResNet pretrained weights
- **Configuration**: 75% accuracy target
- **Location**: `resnet75/`

## 💻 Usage

### Training

#### Train C3D Model:
```bash
cd c3d70
python trainc3d.py
```

#### Train MobileNet Model:
```bash
cd mobilenet73
python trainn73.py
```

#### Train ResNet Model:
```bash
cd resnet75
python train75.py
```

### Testing

#### Test C3D Model:
```bash
cd c3d70
python testc3dd.py
```

#### Test MobileNet Model:
```bash
cd mobilenet73
python test73.py
```

#### Test ResNet Model:
```bash
cd resnet75
python test.75.py
```

### Video Preprocessing

Segment videos into clips:
```bash
python segment_video_30frames.py
```

Augment accident segments:
```bash
python augment_accident_segments.py
```

## 📈 Results

### Model Performance

<!-- Update with your actual results -->
- **C3D**: [Add accuracy and metrics]
- **MobileNet**: 73% accuracy
- **ResNet**: 75% accuracy

### Evaluation Metrics

Each model generates:
- Confusion Matrix
- ROC Curve
- Per-class metrics
- Classification report
- Predictions CSV

Results are saved in respective model directories under `test_results*/` folders.

## 📦 Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch
- torchvision
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Pandas

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

[Add your license information here]

## 👤 Author

**Harsh**
- GitHub: [@harsh-154](https://github.com/harsh-154)

## 🙏 Acknowledgments

- Pretrained models from various sources
- Dataset providers
- Open source community

---

**Note**: Make sure to update the dataset links and add your specific results/metrics in the appropriate sections above.

