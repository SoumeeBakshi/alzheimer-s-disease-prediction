# alzheimer-s-disease-prediction
# Optimizing Alzheimer’s Disease Detection Through Integrated Machine Learning Techniques

This repository contains the implementation and results of a hybrid machine learning approach for early detection of Alzheimer’s Disease (AD), as proposed in our IEEE conference paper.

##  Project Overview

Early and accurate diagnosis of Alzheimer’s Disease is crucial for timely intervention. This project integrates **clinical tabular data** and **MRI imaging data** using a **multimodal hybrid model**. The model combines:
- **Support Vector Machine (SVM)**
- **Random Forest (RF)**
- **Vision Transformer (ViT)**

These models are used in a fusion pipeline to improve prediction accuracy for Alzheimer’s stages.

##  Data

The project uses a multi-modal dataset:
- **Clinical Data**: Includes age, gender, education, memory score, and other features for 430 patients.
- **MRI Images**: T1-weighted brain scans labeled across four cognitive stages (Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented).


##  Methodology

- **Clinical data** → Processed using standard preprocessing techniques, normalized, and used for RF & SVM.
- **MRI images** → Preprocessed, resized, and fed into a fine-tuned ViT model.
- **Feature Fusion** → ViT and RF outputs are concatenated and passed into an SVM classifier for final prediction.

## Technologies Used

- Python
- PyTorch
- Scikit-learn
- NumPy / Pandas
- Matplotlib / Seaborn

## Model Performance

| Model | Accuracy | Precision (AD) | Recall (AD) | F1-Score (AD) |
|-------|----------|----------------|-------------|---------------|
| SVM   | 82%      | 78%            | 70%         | 74%           |
| RF    | 92.56%   | 96%            | 82%         | 89%           |
| ViT   | ~87%     | -              | -           | -             |

ViT performed well on individual image classification, and the hybrid model consistently outperformed standalone models.



