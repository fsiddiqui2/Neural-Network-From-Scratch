import numpy as np

# Loss Function
def BCELoss(y_pred, y):
    assert y_pred.shape == y.shape

    epsilon = 1e-8

    losses = - (y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
    return losses.mean() # y and y_pred are expected to be one dimensional or shape 1 x m

# Activation Function
def sigmoid(X):
    return 1 / (1 + np.exp(-1 * X))

# Model Architecture
class FCNN():
    def __init__(self, n=[2, 3, 3, 1], lr=0.01, seed=42):
        self.rng = np.random.default_rng(seed=seed)
        self.n = n
        self.lr = lr
        self.cache = {}
        self.parameters = {}
        self._init_params()

    def _init_params(self):
        for l in range(1, len(self.n)):
            Wl = self.rng.normal(scale = 0.1, size = (self.n[l], self.n[l-1]))
            bl = np.zeros((self.n[l], 1))

            self.parameters[f"W{l}"] = Wl
            self.parameters[f"b{l}"] = bl
    
    def forward(self, X):
        m = X.shape[1]

        self.cache = {}
        self.cache["A0"] = X

        # forward pass through each layer, with sigmoid activation
        for l in range(1, len(self.n)):
            self.cache[f"Z{l}"] = self.parameters[f"W{l}"] @ self.cache[f"A{l-1}"] + self.parameters[f"b{l}"]
            self.cache[f"A{l}"] = sigmoid(self.cache[f"Z{l}"])
            assert self.cache[f"A{l}"].shape == (self.n[l], m)

        # return the output layer
        return self.cache[f"A{l}"]
        
    def backprop(self, y):
        m = self.cache["A0"].shape[1] # number of observations
        L = len(self.n)-1 # number of layers in model (excluding input)

        gradient = {}
        propagator = None

        # calculate gradient
        for l in range(L, 0, -1):
            if l == L:
                dC_dZl = (1/m)*(self.cache[f"A{l}"] - y) # n[L] x m
                assert dC_dZl.shape == (self.n[l], m)
                propagator = dC_dZl

            else: 
                dC_dAl = self.parameters[f"W{l+1}"].T @ propagator # n[l] x m
                assert dC_dAl.shape == (self.n[l], m)

                Al = self.cache[f"A{l}"]
                dAl_dZl = Al * (1 - Al) # n[l] x m
                assert dAl_dZl.shape == (self.n[l], m)

                dC_dZl = dC_dAl * dAl_dZl # n[l] x m 
                assert dC_dZl.shape == (self.n[l], m)
                propagator = dC_dZl
            
            dC_dWl = dC_dZl @ self.cache[f"A{l-1}"].T 
            assert dC_dWl.shape == (self.n[l], self.n[l-1])

            dC_dbl = np.sum(dC_dZl * 1, axis=1, keepdims=True)
            assert dC_dbl.shape == (self.n[l], 1)
            
            gradient[f"W{l}"] = dC_dWl
            gradient[f"b{l}"] = dC_dbl
        
        # update parameters
        for param in self.parameters:
            self.parameters[param] -= self.lr * gradient[param]


if __name__ == "__main__":
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
    y = np.array([0,1,1,0,0,1,1,0,1,0]).reshape(1, 10)

    # standardize features
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    X = (X - mean)/std

    fcnn = FCNN(lr=0.7, n=[2, 3, 3, 1])

    epochs = 1000
    losses = []
    for epoch in range(1, epochs+1):
        y_pred = fcnn.forward(X)
        loss = BCELoss(y_pred, y)
        losses.append(loss)
        fcnn.backprop(y)
    
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

