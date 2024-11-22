import numpy as np  # for python and numpy's Naive Bayes
from sklearn.naive_bayes import GaussianNB  # for Scikit-learn's Naive Bayes

# Edition 1: use Python and Numpy to implement Gaussian Naive Bayes
class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.class_means = {}
        self.class_stds = {}

    def fit(self, X, y):
        classes = np.unique(y)
        n_samples, n_features = X.shape

        for c in classes:
            X_c = X[y == c]
            self.class_probs[c] = X_c.shape[0] / n_samples
            self.class_means[c] = np.mean(X_c, axis=0)
            self.class_stds[c] = np.std(X_c, axis=0)

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        posteriors = []
        for c in self.class_probs:
            prior = np.log(self.class_probs[c])
            likelihood = np.sum(np.log(self._pdf(c, x)))
            posteriors.append(prior + likelihood)
        return max(range(len(posteriors)), key=lambda i: posteriors[i])

    def _pdf(self, class_label, x):
        mean = self.class_means[class_label]
        std = self.class_stds[class_label]
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)


# Edition 2: use Scikit-learn to implement Gaussian Naive Bayes
def train_naive_bayes_sklearn(train_features, train_labels):
    model = GaussianNB()
    model.fit(train_features, train_labels)
    return model