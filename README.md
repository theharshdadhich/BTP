# Career Success Prediction Using Astrological Data

## Overview
This project explores the application of deep learning in predicting **career success** based on **astrological data**. By simulating astrological factors like planetary positions and zodiac signs, a neural network model is trained to predict whether an individual is likely to experience career success.

The project combines traditional astrological beliefs with modern machine learning techniques, aiming to assess how certain planetary positions and zodiac attributes can be associated with career outcomes.

---

## Data Generation

Since no publicly available dataset provides astrological information with career outcomes, **synthetic data** is generated. The synthetic dataset is created by randomly generating planetary positions and associating them with zodiac signs. The target variable, `career_success`, is also generated randomly, simulating a simplified outcome based on the astrological inputs.

### Dataset Structure

Each data point in the synthetic dataset contains:

- **Planetary Position**: A random integer between 0 and 360 degrees.
- **Zodiac Sign Encoding**: A one-hot encoded vector representing one of the twelve zodiac signs.
- **Career Success**: A binary label (0 for failure, 1 for success).
## Model Architecture
A deep learning model is built using PyTorch, designed as a feedforward neural network to predict career success based on the input features. The model consists of the following layers:

- **Input Layer**: Receives the 13 input features, including the planetary position and zodiac sign encoding.
- **Hidden Layers**: Two fully connected (dense) layers with ReLU (Rectified Linear Unit) activations are used to capture non-linear relationships between the input features and the target variable.
  The first hidden layer has 32 neurons.
  The second hidden layer has 16 neurons.
-**Output Layer**: A single neuron with a sigmoid activation function outputs a probability score between 0 and 1. The model classifies the outcome as career success (1) if the probability is greater than 0.5, and career failure (0) otherwise.
## Training
The model is trained using Binary Cross Entropy Loss (BCELoss), which is suitable for binary classification tasks. The Adam optimizer is used to update the model's weights based on the computed gradients during backpropagation. The model is trained for 100 epochs, with the loss being monitored to ensure that the model improves during training.

## Testing and Evaluation
After training, the model is evaluated on a test set to determine its accuracy. The performance metric used is the accuracy score, which calculates the proportion of correct predictions (career success or failure) over the total predictions made by the model.
