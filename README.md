# Neural Network from Scratch with NumPy
A fully-connected neural network for binary classification built from the ground up with only NumPy. It demonstrates the core mechanics of a neural network, including forward propagation, backpropagation, and gradient descent, without relying on high-level deep learning libraries like TensorFlow or PyTorch.


# About the Files
This repository contains two key files:
- `FCNN.py`: The primary implementation. It refactors the neural network into a flexible and scalable FCNN class.
- `old_reference_code.py`: The original, procedural implementation of the same network. A more direct, script-based approach.

## Features
-  **Object-Oriented Design**: Encapsulates all network logic—parameters, forward pass, and backpropagation—into a clean `FCNN` class.
-  **Scalable Architecture**: Easily change the network's depth and the number of neurons in each layer by modifying the `n` list during class instantiation (e.g., `n=`).
-  **Pure NumPy Implementation**: The entire network is built using only the NumPy library.
-  **Fully Vectorized**: All operations, including the forward and backward passes, are vectorized for high performance, avoiding slow Python `for` loops.
-  **Standard Preprocessing**: Includes feature scaling (standardization) to normalize input data
-  **Training Visualization**: At the end of the training process, the script automatically generates and saves a plot of the training loss over epochs (loss.png).

## Requirements
To run the FCNN.py script, you will need Python 3 and the following libraries:
*   NumPy
*   Matplotlib

You can install these dependencies using `pip`:
```bash
pip install numpy matplotlib
```

## Walkthrough of the `FCNN` Class

### 1. Initialization (`__init__`)
When an `FCNN` object is created, it initializes the network's architecture (`n`), learning rate (`lr`), and random number generator. The `_init_params` helper method creates the weight matrices and bias vectors for each layer with standard initializations (small random normal values for weights, zeros for biases).

### 2. Forward Propagation (`forward`)
This method performs a forward pass. It takes the input data `X` and computes the activations for each layer, caching the intermediate `Z` (pre-activation) and `A` (post-activation) values. These cached values are crucial for the backpropagation step.

### 3. Backpropagation (`backprop`)
This is the core of the learning algorithm, implementing gradient descent. The process works as follows:
1.  **Calculate Gradients**: It iterates backward from the final layer, applying the chain rule to compute the gradient of the loss with respect to every weight and bias in the network. These gradients are stored in a `gradient` dictionary.
2.  **Update Parameters**: After all gradients have been calculated, it iterates through the network's parameters and updates them using the stored gradients and the learning rate.


## To Do
- Test on benchmark dataset (Iris)
- Add convolutional layers and max pooling
