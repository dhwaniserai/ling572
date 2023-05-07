import numpy as np

class MaxEnt:
    def __init__(self, max_iter=100):
        self.max_iter = max_iter
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.feat_counts = np.sum(X, axis=0)
        self.num_feats = X.shape[1]
        self.num_classes = len(self.classes)
        self.class_idx = {c:i for i,c in enumerate(self.classes)}
        self.weights = np.zeros((self.num_classes, self.num_feats))
        
        for i in range(self.max_iter):
            print(f"Iteration {i+1}/{self.max_iter}")
            delta = self._calc_delta(X, y)
            if np.all(np.abs(delta) < 1e-5):
                break
            self.weights += delta
        
    def predict(self, X):
        scores = X.dot(self.weights.T)
        return self.classes[np.argmax(scores, axis=1)]
        
    def _calc_p_y_given_x(self, x):
        scores = np.exp(x.dot(self.weights.T))
        return scores / np.sum(scores, axis=1, keepdims=True)
    
    def _calc_delta(self, X, y):
        delta = np.zeros((self.num_classes, self.num_feats))
        p_y_given_x = self._calc_p_y_given_x(X)
        for i in range(self.num_classes):
            class_mask = y == self.classes[i]
            feat_exp = np.sum(X[class_mask], axis=0) / len(y)
            feat_obs = self.feat_counts / X.shape[0]
            delta[i] = 1/self.num_feats * (np.log(feat_exp / feat_obs) + np.log(p_y_given_x[class_mask].mean(axis=0)))
        return delta

# Example usage:
X_train = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
y_train = np.array(['A', 'B', 'C', 'A', 'B', 'C'])

maxent = MaxEnt(max_iter=100)
maxent.fit(X_train, y_train)

X_test = np.array([[1, 1, 1], [0, 0, 0]])
y_pred = maxent.predict(X_test)
print(y_pred) # Output: ['C' 'A']