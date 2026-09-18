import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from collections import Counter
import time
train_size = 0.8
start_time = time.time()

def data_acquisition():
    iris = fetch_ucirepo(id=53) 
    X = iris.data.features
    y = iris.data.targets['class']
    return X, y

def data_plot(X, y, save = False):
    plot_data = X.assign(class_ = y.squeeze())
    sns.pairplot(plot_data, hue= 'class_', corner=True, plot_kws={'alpha':0.8, 'edgecolor':'black', 's' : 30})
    plt.suptitle('Iris Feature vs Feature Scatter Matrix by Class')
    if save == True:
        plt.savefig('Scatterplots')
    plt.show()

def tts(X,y, train_size, random = 7):
    np.random.seed(random)
    shuffle_indices = np.random.permutation(len(X))
    split_index = int(len(X) * train_size)
    train_indices = shuffle_indices[:split_index]
    test_indices = shuffle_indices[split_index:]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    return X_train, X_test, y_train, y_test

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X):
        X_test = np.array(X)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)
    
    def _predict_single(self, x):
        distance = np.sum((self.X_train - x)**2, axis=1)
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

class ConfusionMatrix:
    def __init__(self, y_test, y_pred, labels = None):
        self.y_pred = list(y_pred)
        self.y_test = list(y_test)
        if len(self.y_pred) != len(self.y_test):
            raise ValueError("y_test and y_pred must have same length")
        if labels == None:
            self.labels = sorted(list(set(y_pred) | set(y_test)))
        else:
            self.labels = labels
        self.num_classes = len(self.labels)
        self.label_to_idx = {label:idx for idx, label in enumerate(self.labels)}
        self.matrix = [[0] *  self.num_classes for _ in range(self.num_classes)]
        self.compute_matrix()

    def compute_matrix(self):
        for pred, true in zip(self.y_pred, self.y_test):
            if pred in self.label_to_idx and true in self.label_to_idx:
                row_idx = self.label_to_idx[true]
                col_idx = self.label_to_idx[pred]
                self.matrix[row_idx][col_idx] += 1
    
    def get_matrix(self):
        return self.matrix
    
    def get_metrics(self):
        metrics = {}
        precisions = []
        recalls = []
        f1_scores = []
        supports = []
        for label, idx in self.label_to_idx.items():
            tp = self.matrix[idx][idx]
            fn = sum(self.matrix[idx]) - tp
            fp = sum(self.matrix[r][idx] for r in range(self.num_classes)) - tp
            tn = len(self.y_pred) - (tp + fn + fp)
            accuracy = (tp + tn)/(tp+tn+fp+fn)
            support = tp + fn
            supports.append(support)
            precision = tp/(tp + fp) if (tp + fp) > 0 else 0
            precisions.append(precision)
            recall = tp/(tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)
            f1_score = (2 * precision * recall)/(precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1_score)
            metrics[label] = {"True Positve: " : tp, "False Positive: " : fp, "True Negative: " : tn, "False Negative " : fn,
            "Accuracy" : f'{accuracy:.4f}',
            "Precision" : f'{precision:.4f}',
            "Recall" : f'{recall:.4f}',
            "F1 Score" : f'{f1_score:.4f}',
            "Support" : support}
        total = len(self.y_test)
        correct = sum(self.matrix[i][i]for i in range(self.num_classes))
        overall_accuracy = correct / total if total > 0 else 0
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1_scores)
        weighted_precision = np.average(precisions,weights=supports)
        weighted_recall = np.average(recalls,weights=supports)
        weighted_f1 = np.average(f1_scores,weights=supports)
        metrics["Overall"] = {
        "Accuracy": f'{overall_accuracy:.4f}',
        "Macro Precision": f'{macro_precision:.4f}',
        "Macro Recall": f'{macro_recall:.4f}',
        "Macro F1 Score": f'{macro_f1:.4f}',
        "Weighted Precision": f'{weighted_precision:.4f}',
        "Weighted Recall": f'{weighted_recall:.4f}',
        "Weighted F1 Score": f'{weighted_f1:.4f}',
        "Total Samples": total
    }
        return metrics

    def __str__(self):
        header = f" " + "".join(f"{str(label):>10}" for label in self.labels)
        divider = "-" * len(header)
        rows = []
        for label, idx in self.label_to_idx.items():
            row_str = f"{str(label):<15}" + "".join(f"{self.matrix[idx][c]:>10}" for c in range(self.num_classes))
            rows.append(row_str)
        return f"{header}\n{divider}\n" + "\n".join(rows)

    def _build_plot(self, cmap, figsize, normalize):
        fig, ax = plt.subplots(figsize=figsize)
        if normalize:
            plot_data = []
            for row in self.matrix:
                row_sum = sum(row)
                if row_sum > 0:
                    plot_data.append([val/row_sum for val in row])
                else:
                    plot_data.append([0] * self.num_classes)
        else:
            plot_data = self.matrix

        sns.heatmap(plot_data, annot=True, cmap=cmap, xticklabels=self.labels, yticklabels=self.labels, square=True, annot_kws={'size':12})
        ax.set_title('Confusion Matrix', fontsize = 7, fontweight = 'bold')
        ax.set_xlabel('Predicted Label', fontsize = 7)
        ax.set_ylabel('True Label', fontsize = 7)
        plt.xticks(rotation = 45, ha = 'right')
        plt.yticks(rotation = 0)
        plt.tight_layout()
        return fig
    
    def plot(self, cmap = 'Blues', figsize=(7, 7), normalize= False):
        self._build_plot(cmap, figsize, normalize)
        plt.show()
    
    def save(self, filename='ConfusionMatrix.png', cmap = 'Blues', figsize=(7, 7), normalize= False):
        fig = self._build_plot(cmap = cmap, figsize=figsize, normalize= normalize)
        fig.savefig(filename)
        plt.close(fig)
        print(f"Plot saved successfully as {filename}.")

def pipeline():
    X, y = data_acquisition()
    X_train, X_test, y_train, y_test = tts(X, y, train_size)
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = ConfusionMatrix(y_test, y_pred, labels=list(y.unique()))
    # print(cm)
    # cm.plot()
    metrics = cm.get_metrics()
    # print(metrics)
    print(f"Accuracy is {float(metrics['Overall']['Accuracy']) * 100}%")
    end_time = time.time()
    data_plot(X, y)
    return metrics, end_time


m, end_time = pipeline()
print(f"Elapsed time = {end_time - start_time}")