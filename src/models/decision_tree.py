import numpy as np
from sklearn.tree import DecisionTreeClassifier

# DT's node class
class Node:
    def __init__(self, gini, feature=None, threshold=None, left=None, right=None, value=None):
        self.gini = gini
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# Python and Numpy edition
class DecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.root = None

    def _gini(self, y):
        classes = np.unique(y)
        n = len(y)
        gini = 1.0 - sum([(np.sum(y == c) / n) ** 2 for c in classes])
        return gini

    def _split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _best_split(self, X, y):
        n_features = X.shape[1]
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                _, y_left, _, y_right = self._split(X, y, feature, threshold)
                gini = (len(y_left) * self._gini(y_left) + len(y_right) * self._gini(y_right)) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature, best_threshold = feature, threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return Node(gini=self._gini(y), value=np.bincount(y).argmax())

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(gini=self._gini(y), value=np.bincount(y).argmax())

        X_left, y_left, X_right, y_right = self._split(X, y, feature, threshold)
        left_node = self._build_tree(X_left, y_left, depth + 1)
        right_node = self._build_tree(X_right, y_right, depth + 1)
        return Node(gini=self._gini(y), feature=feature, threshold=threshold, left=left_node, right=right_node)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)

    def _predict_sample(self, node, sample):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(node.left, sample)
        return self._predict_sample(node.right, sample)

    def predict(self, X):
        return np.array([self._predict_sample(self.root, sample) for sample in X])

# Scikit-learn edition
def train_sklearn_decision_tree(X_train, y_train, max_depth=50):
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion='gini', random_state=42)
    clf.fit(X_train, y_train)
    return clf