# Automated Wound Assessment Using YOLOv8-Seg

This project develops an **automated wound assessment system using YOLOv8-Seg**. The system detects, classifies, and segments wounds from images and estimates the wound area in **cm²**.

## Features

- Wound detection
- Wound severity classification
- Wound segmentation
- Wound area estimation in cm²
- Wound healing monitoring
- Raspberry Pi 5 verification

## Wound Classes

The system classifies wounds into three categories:

- Normal
- Mild
- Severe

## How It Works

**Wound Image → YOLOv8-Seg → Detection & Classification → Segmentation → Wound Area Estimation**

YOLOv8-Seg is used to locate and classify the wound while generating a segmentation mask. The wound area is calculated using the segmented wound and a **1 × 1 cm reference marker** placed near the wound.

## Models

The original YOLOv8-Seg model is also compared with different backbone architectures:

- YOLOv8-Seg
- YOLOv8-Seg + ShuffleNetV2
- YOLOv8-Seg + MobileNetV3-Small
- YOLOv8-Seg + EfficientNetV2-S

## Technologies

- Python
- YOLOv8-Seg
- PyTorch
- OpenCV
- Roboflow
- Google Colab
- Raspberry Pi 5

## Objective

To develop an automated system for **wound detection, severity classification, segmentation, and wound size estimation** using deep learning.

## Author

**Nur Faizah Hambali**
