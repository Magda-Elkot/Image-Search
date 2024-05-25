# Image-Search
# Image Retrieval System

## Introduction
This project aims to implement an image retrieval system using various feature extraction techniques and similarity measures. The system extracts features from images using traditional methods such as histogram-based approaches as well as modern deep learning-based approaches like VGG16 features. It then employs different similarity measures to find the most similar images to a given query image.

## Dataset
The dataset used in this project can be found at https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html.

## Project Structure
The project is organized into several modules:

1. **Feature Extraction Modules**:
    - **Histogram-based Feature Extraction**: This module extracts color histogram features from images.
    - **DCT-based Feature Extraction**: This module extracts low-frequency DCT features from images.
    - **GLCM-based Feature Extraction**: This module extracts texture features using GLCM (Gray-Level Co-occurrence Matrix).
    - **VGG16-based Feature Extraction**: This module extracts high-level features using the VGG16 convolutional neural network.

2. **Similarity Measure Module**:
    - This module calculates the similarity between feature vectors using Euclidean distance.

3. **Combination Module**:
    - This module combines different types of features and applies PCA for dimensionality reduction.

4. **Main Execution Module**:
    - The main module orchestrates the entire process, from reading and processing images to calculating similarities and displaying results.

## Dependencies
Ensure you have the following dependencies installed:
- OpenCV (cv2)
- NumPy
- Matplotlib
- Scikit-learn (for PCA)
- Keras with TensorFlow backend (for VGG16)

## Usage
1. **Configuration**:
    - Modify the file paths according to your dataset directory structure.

2. **Feature Extraction**:
    - Run the feature extraction modules individually or all at once.
    - Ensure the train and test file paths are correctly set for image loading.

3. **Similarity Calculation**:
    - Run the main module to process query images and find the most similar images.
    - Adjust the query image path as needed.

4. **Combining Features**:
    - Optionally, run the combination module to combine different types of features.

5. **Visualization**:
    - Visualize the results, including precision-recall curves and the most similar images.

## Execution
Run the main function to execute the image retrieval system.

```python
if __name__ == "__main__":
    main_combined()
