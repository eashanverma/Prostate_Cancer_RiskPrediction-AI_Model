# Prostate_Cancer_RiskPrediction-AI_Model
## Overview
This project is an **AI-based system for predicting prostate cancer risk from histopathology images**.  
It mirrors the clinical Gleason grading process used by pathologists and classifies cases into **Low, Intermediate, or High Risk** categories.

The pipeline uses:
- **Gleason grade prediction models** to detect the primary and secondary tissue patterns.
- **Fusion-based risk classification** to combine predictions for final risk assessment.
- **Segmentation masks** to focus learning on tumor regions, improving interpretability and accuracy.

The goal is to assist pathologists by providing a consistent, accurate, and transparent AI tool for risk assessment, reducing variability in manual diagnosis.

## Project Objectives
- Automate prostate cancer risk prediction from histopathology slides.
- Replicate the **two-step Gleason grading** process before final risk classification.
- Improve interpretability by following the clinical decision-making workflow.
- Handle **class imbalance** with weighted loss functions.
- Provide **visual performance metrics** including confusion matrices.

## Dataset
- **Source**: [Gleason2019 Challenge Dataset](https://gleason2019.grand-challenge.org/)
- **Contents**: Whole-slide histopathology images with expert-annotated segmentation masks.
- **Preprocessing**:
  - Extracted **RGB patches** and aligned **mask patches**.
  - Generated **4-channel inputs** (RGB + mask) for grade prediction.
  - Created **6-channel fusion inputs** (RGB + grade prediction maps + Gleason score map) for risk classification.
  - Split into **Training (80%)** and **Validation (20%)**, stratified by class.

## Methodology
The pipeline consists of **three stages**:

1. **Primary Grade Detection**  
   - Input: RGB image + segmentation mask.  
   - Output: Primary Gleason grade (3, 4, or 5).  

2. **Secondary Grade Detection**  
   - Same input format as primary grade detection.  
   - Output: Secondary Gleason grade (3, 4, or 5).

3. **Risk Classification (Fusion Model)**  
   - Input: RGB + both grade prediction maps + Gleason score map.  
   - Output: Risk category:  
     - Low Risk (Gleason 0 or 6)  
     - Intermediate Risk (Gleason 7)  
     - High Risk (Gleason 8–10)  

## Features
- **Segmentation-guided learning**: Models focus on tumor regions.
- **Multi-stage architecture**: Mimics human grading workflow.
- **Class-weighted loss functions**: Handles imbalanced datasets.
- **Visual explainability**: Confusion matrices for model interpretation.
- **Clinically relevant outputs**: Aligns with how pathologists assess risk.
```bash
pip install -r requirements.txt
