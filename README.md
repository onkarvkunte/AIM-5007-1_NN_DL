# AIM-5007-1_NN_DL

# Optimizing Traffic Sign Recognition: Custom CNN vs. Pretrained Models

This project investigates the effectiveness of custom Convolutional Neural Networks (CNNs) compared to pretrained models for traffic sign recognition. The goal is to improve the accuracy and reliability of traffic sign detection systems, which are critical for autonomous vehicle safety.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Datasets](#datasets)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Challenges](#challenges)
- [Future Work](#future-work)
- [Contributors](#contributors)

## Introduction

The rapid development of autonomous vehicles has amplified the need for accurate traffic sign recognition systems. This project explores modern deep learning techniques to address the challenges posed by varying traffic signs, lighting conditions, and geographical differences.

Our research benchmarks a custom CNN model against widely used pretrained architectures such as ResNet, EfficientNet, and DenseNet.

## Features

- Developed a custom CNN model tailored for traffic sign recognition.
- Evaluated and fine-tuned pretrained models, including ResNet50, DenseNet121, and EfficientNetB0.
- Utilized the German Traffic Sign Recognition Benchmark (GTSRB) dataset.
- Achieved over 99% accuracy with the custom CNN model.
- Compared performance metrics, including training and validation loss, across models.

## Datasets

### German Traffic Sign Recognition Benchmark (GTSRB)
- **Total Images**: 51,839
- **Classes**: 43 unique traffic signs
- **Key Features**:
  - Diverse lighting conditions, angles, and physical variations.
  - Widely used for traffic sign recognition benchmarking.

### Other Datasets Explored
- Belgium Traffic Sign Dataset (BTSD)
- Chinese Traffic Sign Database (TSRD)

| Dataset | Total Images | Classes |
|---------|--------------|---------|
| GTSRB  | 51,839       | 43      |
| BTSD   | 7,095        | 62      |
| TSRD   | 6,164        | 58      |

## Model Architectures

### Custom CNN
- **Architecture**:
  - Six convolutional layers for feature extraction.
  - Batch normalization and ReLU activation.
  - Dropout layers to prevent overfitting.
  - Dense layers for classification.
- **Training Setup**:
  - Input dimensions: `3x50x50`
  - Optimizer: Adam with learning rate scheduling.
  - Loss Function: Cross-Entropy Loss.

### Pretrained Models
- **EfficientNetGTSRB**: Fine-tuned EfficientNetB0 for traffic sign recognition.
- **ResNetGTSRB**: Utilized ResNet50 for its deep architecture and skip connections.
- **DenseNetGTSRB**: Leveraged DenseNet121 for feature reuse and efficiency.

## Results

| Model       | Test Accuracy | Batch Size |
|-------------|---------------|------------|
| Custom CNN  | 99.17%        | 64         |
| DenseNet121 | 98.06%        | 64         |
| EfficientNetB0 | 97.82%     | 64         |
| ResNet50    | 95.50%        | 64         |
| MobileNetV2 | 97.43%        | 64         |
| VGG16       | 59.40%        | 64         |

### Key Insights
- The custom CNN outperformed all pretrained models, achieving the highest accuracy.
- EfficientNet and DenseNet offered competitive results with smaller model sizes.

## Challenges

- **Data Imbalance**: Addressed by using data augmentation techniques such as rotation and resizing.
- **Model Overfitting**: Mitigated using dropout layers and learning rate scheduling.
- **Computational Constraints**: Optimized models for efficient training on limited hardware.

## Future Work

- Extend the model to recognize international traffic signs for broader applicability.
- Integrate real-time inference capabilities for on-road applications.
- Explore additional datasets to improve model generalization.

## Contributors

- **Onkar Kunte** (Yeshiva University) - [okunte@mail.yu.edu](mailto:okunte@mail.yu.edu)  
- **Shashidhar Reddy** (Yeshiva University) - [sainala@mail.yu.edu](mailto:sainala@mail.yu.edu)

---

This README provides an overview of your project and its core findings. Let me know if you’d like additional visuals or examples added!
