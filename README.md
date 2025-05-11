# Deepfake Image Detection System

This project focuses on developing a **robust and scalable system for detecting deepfake images** using state-of-the-art machine learning techniques. It explores the use of Convolutional Neural Networks (CNNs), ensemble learning methods, and Generative Adversarial Networks (GANs) to address the growing threat posed by manipulated digital media.

## Problem Statement

The rapid advancement and proliferation of deepfake technology, particularly through the use of GANs, have led to an alarming increase in hyper-realistic fake images. These manipulated visuals are being exploited for malicious activities such as misinformation campaigns, character defamation, and cybercriminal acts, posing a serious threat to the authenticity of digital media. Existing detection systems often struggle with the evolving sophistication of fake generation methods, generalizability, and robustness when exposed to diverse or high-quality fakes. This project aims to bridge this gap by developing an efficient and scalable detection system.

## Objectives

The primary objective is to contribute an effective, accurate, and adaptable fake image detection system. Specific goals include:
*   **Develop a High-Accuracy Detection System:** Leverage CNNs and other advanced machine learning techniques to accurately distinguish real images from deepfakes.
*   **Adaptability to Various Deepfake Techniques:** Ensure the system can generalize across different types of GAN-generated images and is not restricted to specific datasets or manipulation patterns.
*   **Feature Extraction and Analysis:** Utilize deep learning architectures to extract meaningful image features that indicate subtle manipulations.
*   **Practical Real-World Integration:** Design the system to be applicable in real-world contexts such as media forensics, social media monitoring, and cybersecurity.
*   **Support for Media Authenticity:** Provide a tool that supports the verification of digital media, enhancing trust in online content.

## Motivation

The project is motivated by the **growing threat of deepfake technology being used for malicious purposes**, including tarnishing reputations of public figures and enabling cybersecurity crimes like fraud and impersonation. The danger extends to finance, national security, and journalism. The goal is to provide a reliable, efficient, and easy-to-use detection system to combat deepfakes and contribute to a safer, more secure digital environment where people can trust online images.

## Key Features and Components

The system incorporates several key components and techniques:
*   **CNN-based Detection:** Training and evaluation of various CNN architectures, including ResNet50, VGG16, DenseNet121, XceptionNet, and EfficientNet variants (B0, B2, B4) for deepfake image classification.
*   **Data Preprocessing:** Comprehensive preprocessing pipeline including resizing, normalization, label encoding, data splitting (stratified for Yonsei), and data augmentation (for 140k dataset).
*   **Ensemble Learning:** Implementation of techniques like Averaging Probability, Majority Voting, Weighted Majority Voting, and Stacking to combine the predictions of top-performing CNN models for improved accuracy and robustness, particularly on smaller datasets.
*   **GAN Architecture:** Integration of a StyleGAN2-ADA-based generator for creating synthetic fake faces and a custom hybrid CNN discriminator to distinguish real from GAN-generated images.
*   **Evaluation Metrics:** Use of standard classification metrics (Accuracy, Precision, Recall, F1-Score) and generative model evaluation metrics (Fréchet Inception Distance - FID) to assess performance.

## Datasets Used

The project was trained and evaluated using two main datasets:
*   **Yonsei Deepfake Image Dataset:** Contains 2041 images (960 fake, 1081 real). Used for initial evaluation and ensemble learning experiments. Its smaller size and limited diversity presented challenges like overfitting.
*   **NVIDIA Flickr Dataset (140k subset):** Comprises 140,000 images (70k real from Flickr, 70k fake from StyleGAN). Used for training and evaluating individual CNN models and the GAN components. This dataset's larger size enabled better generalization and improved accuracy.

## Key Technologies and Tools

*   **Python Environment:** Anaconda for package management.
*   **Machine Learning Libraries:** TensorFlow/Keras and PyTorch for model building and training.
*   **Computing Resources:** NVIDIA GPUs with CUDA Toolkit (locally and in cloud), Google Colab (TPU v4), and Kaggle Notebooks (NVIDIA Tesla P100).
*   **Development Tools:** Jupyter Notebook for experimentation, ImageDataGenerator and Albumentations for data augmentation.
*   **Evaluation Tools:** Scikit-learn for data splitting and metrics, NumPy for prediction aggregation, Python's time module for timing, InceptionV3 for FID calculation.
*   **Explainability (Future Scope):** SHAP and LIME, CAMs and Grad-CAM.

## Architecture Overview

The system architecture follows a layered approach:
1.  **Data Pipeline Layer:** Handles data ingestion, preprocessing, and organization for training and evaluation.
2.  **Model Training and Processing Layer:** Implements and optimizes CNN models, ensemble methods, and the GAN framework, leveraging high-performance computing.
3.  **Evaluation and Output Layer:** Assesses model performance using quantitative metrics and qualitative visualizations.

## Key Results Highlights

*   On the Yonsei dataset, individual CNN accuracy was around 63% (ResNet50 highest).
*   Ensemble learning on the Yonsei dataset significantly improved performance, with **Stacking using PCA and SVC achieving a maximum accuracy of 72%**.
*   On the larger NVIDIA Flickr dataset, models performed significantly better. DenseNet121 achieved **92% accuracy and a 98.43% F1-score**. VGG16 achieved **95% ROC-AUC**.
*   The GAN architecture achieved an **overall discriminator accuracy of 73%** and a FID score of 51.12.

## Key Challenges Faced

*   **Training Time and Computation:** Training deep CNNs on large datasets required significant computational power and time, necessitating the use of GPUs and cloud resources.
*   **Hardware Limitations:** Local hardware often lacked the necessary GPU resources, overcome by using cloud platforms.
*   **Overfitting on Smaller Dataset:** Models trained on the Yonsei dataset showed overfitting due to limited size and diversity, addressed with data augmentation and regularization techniques.
*   **Dataset Imbalance:** Imbalanced class distribution in initial stages caused biased predictions, mitigated by balancing strategies and class weights.
*   **Longer Evaluation Time for Larger Models:** Evaluating large models and computing metrics like FID was time-intensive, managed with batch-wise evaluations and cached feature extraction.
*   **Integration and Environment:** Managing dependencies and ensuring compatibility across different tools and environments (Kaggle, Colab, local) required careful configuration and modularization.

## Future Scope

Future directions for the project include:
*   **Real-Time Inference:** Implementing optimized models for real-time deployment using containerization (Docker/Kubernetes) and APIs.
*   **Explainability and Trust:** Integrating XAI methods like SHAP, LIME, and CAMs to provide interpretable model decisions.
*   **Broader Applications:** Expanding detection to other modalities (audio, video, text) and developing a multi-modal system for integration into digital forensics, journalism, and cybersecurity.

## Setup

To run this project, you will need Python and the dependencies listed in the `requirements.txt` file.

Clone the repository:
```bash
git clone <repository_url>
cd my-project
```

Install dependencies:
```bash
pip install -r requirements.txt
```


