## Hand Gesture Recognition
This project focuses on hand gesture recognition using deep learning and embedding-based machine learning approaches.

The study compares pretrained convolutional neural network models and a classical machine learning baseline for an 18-class hand gesture classification task. The experiments include frozen backbone training, partial fine-tuning, feature extraction, PCA dimensionality reduction, SVM classification, and robustness analysis under Gaussian blur corruption.

## Dataset
The project uses the HaGRID Classification 512p dataset, which contains RGB images of static hand gestures collected under diverse real-world conditions.

A class-balanced subset was created by randomly sampling 5,555 images from each of the 18 gesture classes, resulting in 99,990 total images. The dataset was then split into train, validation, and test sets using a stratified 70/15/15 split.

## Models and Methods
The following approaches were implemented and evaluated:

ResNet-50 with frozen backbone

ResNet-50 with partial fine-tuning

EfficientNet-B0 with frozen backbone

EfficientNet-B0 with partial fine-tuning

SVM classifier trained on PCA-reduced ResNet-50 feature embeddings

## Technologies Used
Python

PyTorch

torchvision

timm

scikit-learn

NumPy

Matplotlib

Google Colab / Jupyter Notebook
## Results
The best-performing model was the SVM classifier trained on ResNet-50 embeddings, achieving 97.67% test accuracy and a macro F1-score of 0.9767.

The fine-tuned ResNet-50 model also achieved strong performance with 97.16% test accuracy. Robustness experiments showed that the embedding-based SVM approach was the most stable under Gaussian blur corruption.
