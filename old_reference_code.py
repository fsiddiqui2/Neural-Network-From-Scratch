import numpy as np

# set random seed
rng = np.random.default_rng(seed=42)

# network size by layer
n = [2, 3, 3, 1]
# number of observations
m = 10

# dummy data with 2 features and 10 observations
X = np.array([
    [150, 70],
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
]).T

# dummy labels, either 0 or 1
y = np.array([0,1,1,0,0,1,1,0,1,0]).reshape(n[-1], m)

# initialize weights and biases
W1 = rng.normal(scale = 0.1, size = (n[1], n[0]))
b1 = np.zeros((n[1], 1))

W2 = rng.normal(scale = 0.1, size = (n[2], n[1]))
b2 = np.zeros((n[2], 1))

W3 = rng.normal(scale = 0.1, size = (n[3], n[2]))
b3 = np.zeros((n[3], 1))

parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

# forward pass
def forward(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = W1 @ X + b1 # b is automaticaly broadcasted from (n, 1) to (n, m)
    A1 = sigmoid(Z1)
    assert A1.shape == (n[1], m)

    Z2 = W2 @ A1 + b2 
    A2 = sigmoid(Z2)
    assert A2.shape == (n[2], m)

    Z3 = W3 @ A2 + b3 
    A3 = sigmoid(Z3)
    assert A3.shape == (n[3], m)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3, "A0": X}
    
    return A3, cache

# Loss function
def BCELoss(y_pred, y):
    assert y_pred.shape == (1, m) and y.shape == (1, m)

    epsilon = 1e-8

    losses = - (y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
    return losses.mean() # y and y_pred are expected to be one dimensional or shape 1 x m

# Activation Function
def sigmoid(X):
    return 1 / (1 + np.exp(-1 * X))

# Backpropagation
def backprop(y_pred, y, cache, parameters):
    A0, A1, A2, A3 = cache["A0"], cache["A1"], cache["A2"], cache["A3"]
    W1, W2, W3 = parameters["W1"], parameters["W2"], parameters["W3"]

    m = A0.shape[1]

    # dC_dA3 = (-1/m)*((y / A3) - (1 - y)/(1 - A3)) # n3 x m
    # dA3_dZ3 = A3 * (1 - A3) # n3 x m
    # dC_dZ3 = dC_dA3 * dA3_dZ3 # n3 x m 
    dC_dZ3 = (1/m)*(A3 - y) # n3 x m 
    # dZ3_dW3 = A2 # n2 x m
    dC_dW3 = dC_dZ3 @ A2.T # n3 x n2
    dC_db3 = np.sum(dC_dZ3 * 1, axis=1, keepdims=True) # n3 x 1

    # dZ3_dA2 = W3 # n3 x n2

    dC_dA2 = W3.T @ dC_dZ3 # n2 x m
    dA2_dZ2 = A2 * (1 - A2) # n2 x m
    dC_dZ2 = dC_dA2 * dA2_dZ2 # n2 x m 
    # dZ2_dW2 = A1 # n1 x m
    dC_dW2 = dC_dZ2 @ A1.T # n2 x n1
    dC_db2 = np.sum(dC_dZ2 * 1, axis=1, keepdims=True) # n2 x 1

    # dZ2_dA1 = W2 # n2 x n1

    dC_dA1 = W2.T @ dC_dZ2 # n1 x m
    dA1_dZ1 = A1 * (1 - A1) # n1 x m
    dC_dZ1 = dC_dA1 * dA1_dZ1 # n1 x m 
    # dZ1_dW1 = A0 # n0 x m
    dC_dW1 = dC_dZ1 @ A0.T # n1 x n0
    dC_db1 = np.sum(dC_dZ1 * 1, axis=1, keepdims=True) # n1 x 1

    gradient = {"W1": dC_dW1, "b1": dC_db1, "W2": dC_dW2, "b2": dC_db2, "W3": dC_dW3, "b3": dC_db3}
    return gradient

# standardize features
mean = np.mean(X, axis=1, keepdims=True)
std = np.std(X, axis=1, keepdims=True)
std = np.where(std == 0, 1, std)
X = (X - mean)/std
# print(X)
# print(mean)
# print(std)


# training loop
lr = 0.7
epochs = 1000
losses = []
for epoch in range(1, epochs+1):
    y_pred, cache = forward(X, parameters)
    loss = BCELoss(y_pred, y)
    losses.append(loss)
    #print(f"Epoch {epoch}: {loss}")

    gradient = backprop(y_pred, y, cache, parameters)
    for param in parameters:
        parameters[param] -= lr * gradient[param]

print("Probabilities: ", y_pred)
print("Prediction: ", np.where(y_pred > .5, 1, 0))
print("Truth:      ", y)
print("Loss: ", loss)

import matplotlib.pyplot as plt
plt.plot(range(1, epochs+1), losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.png")

