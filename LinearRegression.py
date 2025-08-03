import numpy as np
class LinearRegression:

    def __init__(self,iterations=1000,learning_rate=0.01):
        self.iter = iterations
        self.alpha = learning_rate

    def fit(self,X,y):
        
        self.X=X
        
        self.Y = y
        
        self.rows, self.cols = X.shape
        
        self.m = np.zeros(self.cols)
        
        self.c = 0
        
        for i in range(self.iter):
            self.update_weights()
        return self
    
    def update_weights(self):
        y_pred = self.predict(self.X)
        
        dm = 2*((self.X.T).dot(y_pred - self.Y))/self.rows
        dc = 2*np.sum(y_pred - self.Y)/self.rows
        self.m -= self.alpha * dm
        self.c -= self.alpha * dc
        return self
    def predict(self,X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.cols:
            raise ValueError("Input features must match the training data dimensions.")
        return np.dot(X, self.m) + self.c
    
    def score(self,X,y):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.cols:
            raise ValueError("Input features must match the training data dimensions.")
        
        y_pred = self.predict(X)
        error = y_pred - y
        rmse = np.sqrt(np.mean(error ** 2/self.rows))
        return rmse
    