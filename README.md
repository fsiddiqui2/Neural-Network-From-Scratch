# Neural Network from Scratch with NumPy
A fully-connected neural network for binary classification built from the ground up with only NumPy. It demonstrates the core mechanics of a neural network, including forward propagation, backpropagation, and gradient descent, without relying on high-level deep learning libraries like TensorFlow or PyTorch.

## Features
- Pure NumPy Implementation: The entire network is built using only the NumPy library.
- Fully Vectorized: All operations, including the forward and backward passes, are vectorized for high performance, avoiding slow Python for loops.
- Standard Preprocessing: Includes feature scaling (standardization) to normalize input data
- Training Visualization: At the end of the training process, the script automatically generates and saves a plot of the training loss over epochs (loss.png).

## Requirements
To run this script, you will need Python 3 and the following libraries:
*   NumPy
*   Matplotlib

You can install these dependencies using `pip`:
```bash
pip install numpy matplotlib
```

## To Do
- Customizable architecture
- Test on benchmark dataset (Iris)
- Add convolutional layers and max pooling
