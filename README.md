# Handwritten-digit-recognition-using-cnn
CNN-based handwritten digit recognition system that accurately classifies digits (0–9) using the MNIST dataset.

# Project Overview

The model is trained on handwritten digit images and learns to extract important features such as edges and shapes automatically. The system can predict digits from both test data and user-drawn input through a GUI.

# Features
 - CNN model built using PyTorch
 - Training and testing pipeline
 - Accuracy evaluation
 - Visualization of predictions
 - Interactive GUI for digit drawing and prediction
 - Real-time inference

# Tech Stack
- Python
- PyTorch
- NumPy
- Matplotlib
- Tkinter
- PIL


**Project Structure**

├── train.py                  # Model training

├── predict.py                # Accuracy evaluation

├── visualize_predictions.py  # Visualize predictions

├── draw_predict.py           # GUI for drawing digits

├── models/
│   └── cnn_model.py          # CNN architecture

├── data/
│   └── load_data.py          # Data loading

├── digit_cnn_model.pth       # Trained model



# How It Works
Train the Model

The CNN is trained using training data with loss optimization and backpropagation.

python train.py
Evaluating Accuracy

The trained model is tested on unseen data to measure performance.

python predict.py
Visualizing Predictions

Displays sample predictions along with actual labels.

python visualize_predictions.py
Real-Time Digit Prediction (GUI)

Draw a digit on the canvas and let the model predict it instantly.

python draw_predict.py
Model Architecture
Convolutional Layers (feature extraction)
ReLU Activation
Pooling Layers
Fully Connected Layers
Softmax Output Layer
Results

The model achieves high accuracy on test data and performs well on user-drawn digits, demonstrating the effectiveness of CNNs in image classification tasks.

# Applications

- Postal code recognition
- Bank check processing
- Optical Character Recognition (OCR)
- Digit-based automation systems

  
# Key Learnings
- Understanding CNN architecture
- Image preprocessing techniques
- Model training and evaluation in PyTorch
- Real-time prediction using GUI integration
- Future Improvements
- Improve accuracy with deeper architectures
- Add support for multi-digit recognition
- Deploy as a web application
- Use real-world datasets
