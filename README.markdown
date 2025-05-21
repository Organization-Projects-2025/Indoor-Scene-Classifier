# Interior Room Classification Project

## Project Overview

This project was developed as part of my Machine Learning course during my 3rd year of Computer Science. It’s a machine learning solution designed to integrate with my Real-Estate MERN (MongoDB, Express, React, Node.js) project. The goal is to automate the classification of interior room images uploaded by property sellers. Instead of manually labeling each image, the model processes them and groups them into categories: bathroom, bedroom, dining room, kitchen, and living room. This enhances the efficiency of the property listing process in the real-estate application.

## Technologies Used

- **Programming Language**: Python
- **Machine Learning Libraries**:
  - TensorFlow: For building and training deep learning models like CNNs, MobileNet, and ResNet.
  - scikit-learn: For implementing traditional algorithms like SVM and Logistic Regression with GridSearchCV.
  - Gradio: For creating an interactive web interface to showcase predictions.
- **Other Libraries**:
  - pandas: For data handling.
  - NumPy: For numerical operations.
  - Pillow: For image processing.

## Dataset

The dataset contains **12,335 images** across five classes:

- Bath
- Bed
- Dining Room
- Kitchen
- Living Room

_Note: The dataset is not included in this repository due to its large size and the time required to upload it._

## Models Implemented

I implemented and evaluated multiple models to classify the room images. Below are the models and their performance metrics (precision, recall, F1-score) based on a test set of 2,467 images.

### 1. CNN + SVM (with GridSearchCV)

- **Accuracy**: 90%
- **Classification Report**:
  ```
         Class       Precision    Recall    F1-Score    Support
        bath         0.92        0.95      0.94        486
         bed         0.91        0.90      0.91        489
  dining room         0.90        0.88      0.89        521
     kitchen         0.90        0.87      0.88        447
  living room         0.85        0.88      0.87        524
  ```

### 2. CNN + LSTM

- **Accuracy**: 86%
- **Classification Report**:
  ```
         Class       Precision    Recall    F1-Score    Support
        bath         0.84        0.95      0.89        486
         bed         0.86        0.90      0.88        489
  dining room         0.90        0.78      0.84        521
     kitchen         0.86        0.86      0.86        447
  living room         0.86        0.85      0.85        524
  ```

### 3. Custom CNN

- **Accuracy**: 82%
- **Classification Report**:
  ```
         Class       Precision    Recall    F1-Score    Support
        bath         0.86        0.90      0.88        486
         bed         0.75        0.88      0.81        489
  dining room         0.84        0.77      0.81        521
     kitchen         0.88        0.80      0.83        447
  living room         0.80        0.76      0.78        524
  ```

### 4. Logistic Regression + CNN (with GridSearchCV)

- **Accuracy**: 89%
- **Classification Report**:
  ```
         Class       Precision    Recall    F1-Score    Support
        bath         0.93        0.94      0.94        486
         bed         0.90        0.89      0.90        489
  dining room         0.90        0.87      0.88        521
     kitchen         0.91        0.88      0.90        447
  living room         0.83        0.88      0.85        524
  ```

### 5. MobileNet

- **Accuracy**: 90%
- **Classification Report**:
  ```
         Class       Precision    Recall    F1-Score    Support
        bath         0.93        0.95      0.94        486
         bed         0.93        0.89      0.91        489
  dining room         0.89        0.90      0.90        521
     kitchen         0.92        0.87      0.89        447
  living room         0.86        0.90      0.88        524
  ```

### 6. ResNet

- **Accuracy**: 93%
- **Classification Report**:
  ```
         Class       Precision    Recall    F1-Score    Support
        bath         0.95        0.98      0.96        486
         bed         0.95        0.94      0.94        489
  dining room         0.92        0.90      0.91        521
     kitchen         0.95        0.91      0.93        447
  living room         0.88        0.91      0.89        524
  ```

## Gradio Interface

I built an interactive interface using Gradio to demonstrate the models’ capabilities. Users can:

- Choose one of the pre-trained models.
- Upload an image.
- Get the predicted room label along with additional statistics.

This interface is a proof-of-concept for how the classification system could work in the real-estate application.

## Challenges Faced

- **Class Imbalance**: Some classes had fewer images, which affected model performance. I addressed this by adjusting class weights during training.
- **Model Tuning**: Finding the right hyperparameters was tricky, especially for the custom CNN. GridSearchCV helped optimize the SVM and Logistic Regression models.
- **Overfitting**: Deeper models like ResNet showed overfitting initially, which I mitigated with dropout and regularization.

## Setup and Installation

To run this project locally:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Organization-Projects-2025/Indoor-Scene-Classifier.git
   cd Interior-Room-Classifier
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   _Note: Ensure you have Python 3.8+ installed._

3. **Run the Gradio Interface**:

   ```bash
   python gradio_app.py
   ```

4. **Access the Interface**:
   - Open your browser and go to the provided URL (e.g., `http://127.0.0.1:7860`).
   - Select a model, upload an image, and see the prediction.

_Note: Since the dataset isn’t uploaded, you’ll need to provide your own images or request access to the original dataset for full functionality._
