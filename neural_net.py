import numpy as np

# actual brain of the project

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class CoffeeNeuralNet:

    def __init__(self):
        np.random.seed(42) 

        # hidden layer, connects 5 inputs to 8 hidden neurons
        self.W1 = np.random.randn(5, 8) * 0.1
        self.b1 = np.zeros(8)

        # output layer, connects 8 hidden neurons to 1 output (revenue)
        self.W2 = np.random.randn(8, 1) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, X):
        # forward pass
        self.z1 = X @ self.W1 + self.b1 # z1 shape: (batch, 8)
        self.a1 = relu(self.z1) # activated hidden layer

        self.z2 = self.a1 @ self.W2 + self.b2 # z2 shape: (batch, 1)
        return self.z2 # this is our predicted revenue

    def train(self, X, y, learning_rate=0.01, epochs=1000):
        # training loop
        y = y.reshape(-1, 1)
        losses = []

        for epoch in range(epochs):

            # FORWARD PASS
            predictions = self.forward(X)

            # LOSS
            error = predictions - y
            loss = np.mean(error ** 2)
            losses.append(loss)

            # BACKWARD PASS

            m = X.shape[0] # number of training samples

            # gradient for output layer
            d_output = (2 / m) * error # derivative of MSE loss
            dW2 = self.a1.T @ d_output # how much W2 contributed
            db2 = np.sum(d_output, axis=0)

            # gradient for hidden layer
            d_hidden = d_output @ self.W2.T # push gradient back
            d_hidden *= relu_derivative(self.z1) # apply ReLU's derivative
            dW1 = X.T @ d_hidden
            db1 = np.sum(d_hidden, axis=0)

            # UPDATE
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

            if (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}   Loss: {loss:.4f}")

        return losses

    def predict(self, X):
        return self.forward(X)
