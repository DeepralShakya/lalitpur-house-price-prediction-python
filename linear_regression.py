import numpy as np


class LinearRegressionScratch:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None

    def fit(self, X, y, learning_rate=0.01, epochs=1000, early_stopping=True, tol=1e-4):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = (X - self.mean) / (self.std + 1e-8)  # Normalize features

        n_samples, n_features = X.shape  # initialize weight
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

        prev_loss = np.inf
        no_improvement_epochs = 0

        for epoch in range(epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            loss = np.mean((y_pred - y) ** 2)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = np.mean(y_pred - y)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            if early_stopping:
                if abs(prev_loss - loss) < tol:
                    no_improvement_epochs += 1
                    if no_improvement_epochs >= 10:
                        print(f"Early Stopping at epoch {epoch}. Loss not improving significantly.")
                        break
                else:
                    no_improvement_epochs = 0
                prev_loss = loss

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before `predict`.")
        X = (X - self.mean) / (self.std + 1e-8)
        predictions = np.dot(X, self.weights) + self.bias

        return predictions

    def get_weights(self):
        return self.weights, self.bias
