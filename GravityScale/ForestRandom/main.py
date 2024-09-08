import pandas as pd
import numpy as np
import math

from collections import Counter

from sklearn.model_selection import KFold


def entropy(labels):
    label_counts = Counter(labels)
    total_labels = len(labels)
    entropy_value = 0.0

    for label_count in label_counts.values():
        probability = label_count / total_labels
        entropy_value -= probability * math.log2(probability)

    return entropy_value

def information_gain(parent_labels, left_labels, right_labels):
    parent = entropy(parent_labels)
    left = entropy(left_labels)
    right = entropy(right_labels)
    
    left_weight = len(left_labels) / len(parent_labels)
    right_weight = len(right_labels) / len(parent_labels)
    
    return parent - (left_weight * left + right_weight * right)

def best_split(data, labels):
    best_gain = 0
    best_attribute = None
    best_split_point = None
    n_features = data.shape[1]

    for feature in range(n_features):
        unique_values = np.unique(data[:, feature])
        for split_point in unique_values:
            left_indices = data[:, feature] <= split_point
            right_indices = data[:, feature] > split_point

            left_labels = labels[left_indices]
            right_labels = labels[right_indices]
            
            gain = information_gain(labels, left_labels, right_labels)
            
            if gain > best_gain:
                best_gain = gain
                best_attribute = feature
                best_split_point = split_point

    return best_attribute, best_split_point

class RandomForest:
    def __init__(self, n_trees, max_features):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, data, labels):
        for _ in range(self.n_trees):
            indices = np.random.choice(len(data), len(data), replace=True)

            subset_data = data[indices]
            subset_labels = labels[indices]

            tree = build_tree(subset_data, subset_labels, self.max_features)

            self.trees.append(tree)

    def predict_tree(self, tree, sample):
        if not isinstance(tree, tuple):
            return tree
        
        attribute, threshold, left_tree, right_tree = tree

        if sample[attribute] <= threshold:
            return self.predict_tree(left_tree, sample)
        else:
            return self.predict_tree(right_tree, sample)
    
    def predict(self, data):
        predictions = []

        for sample in data:
            tree_predictions = [self.predict_tree(tree, sample) for tree in self.trees]
            most_common = Counter(tree_predictions).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

    def accuracy(self, predictions, labels):
        return np.sum(predictions == labels) / len(labels)

def build_tree(data, labels, num_features=None):
    if len(set(labels)) == 1:
        return labels[0]
    
    if num_features is None:
        num_features = data.shape[1]
    
    features_indices = np.random.choice(data.shape[1], num_features, replace=False)
    
    best_gain = 0
    best_attribute = None
    best_split_point = None
    
    for feature in features_indices:
        unique_values = np.unique(data[:, feature])
        for split_point in unique_values:
            left_indices = data[:, feature] <= split_point
            right_indices = data[:, feature] > split_point
            left_labels = labels[left_indices]
            right_labels = labels[right_indices]
            
            gain = information_gain(labels, left_labels, right_labels)
            
            if gain > best_gain:
                best_gain = gain
                best_attribute = feature
                best_split_point = split_point

    if best_gain == 0:
        return Counter(labels).most_common(1)[0][0]
    
    left_indices = data[:, best_attribute] <= best_split_point
    right_indices = data[:, best_attribute] > best_split_point
    
    left_tree = build_tree(data[left_indices], labels[left_indices], num_features)
    right_tree = build_tree(data[right_indices], labels[right_indices], num_features)
    
    return (best_attribute, best_split_point, left_tree, right_tree)

def k_fold_cross_validation(data, labels, k=5, trees=10, max_features=None):
    kf = KFold(n_splits=k)
    accuracies = []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        forest = RandomForest(n_trees=trees, max_features=max_features)
        forest.fit(X_train, y_train)

        predictions = forest.predict(X_test)

        accuracy = forest.accuracy(predictions, y_test)
        accuracies.append(accuracy)

        print(f'Acurácia da amostra: {accuracy * 100:.2f}%')
    
    return np.mean(accuracies)

data = pd.read_csv('dados_treinamento.csv')

features = data[['qualidade_pressao_arterial', 'pulso', 'respiracao']].values
labels = data['rotulo'].values

average_accuracy = k_fold_cross_validation(features, labels, k=5, trees=2, max_features=int(np.sqrt(features.shape[1])))
print(f'Acurácia média: {average_accuracy * 100:.2f}%')