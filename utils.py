import numpy as np
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as FLD
from sklearn.manifold import TSNE
import umap

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score



######################### Statistics #############################

# Compute the mean for a single feature
def stats_mean(data: np.ndarray) -> float:
    return np.mean(data)

# Compute the standard deviation for a single feature
def stats_SD(data: np.ndarray) -> float:
    return np.std(data)

# Compute the variance for a single feature
def stats_variance(data: np.ndarray) -> float:
    return np.var(data)

# Compute the minimum value for a single feature
def stats_min(data: np.ndarray) -> float:
    return np.min(data)

# Compute the maximum value for a single feature
def stats_max(data: np.ndarray) -> float:
    return np.max(data)

# Compute the interquartile range for a single feature
def stats_interquartile_range(data: np.ndarray) -> float:
    return stats.iqr(data)

# Compute the quantiles for a single feature
def stats_quantiles(data: np.ndarray, partition: list) -> dict:
    values = np.percentile(data, partition)
    return {q: v for q, v in zip(partition, values)}

# Compute the skewness for a single feature
def stats_skew(data: np.ndarray) -> float:
    return stats.skew(data)

# Compute the kurtosis for a single feature
def stats_kurtosis(data: np.ndarray) -> float:
    return stats.kurtosis(data)
##################################################################


######################### Projections ############################
def proj_PCA(data: np.ndarray, n: int) -> np.ndarray:
    return PCA(n_components=n).fit_transform(data)

def proj_FLD(data: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    max_components = len(set(labels)) - 1
    n = max(1, min(n, max_components))
    return FLD(n_components=n).fit_transform(data, labels)

def proj_TSNE(data: np.ndarray, n: int) -> np.ndarray:
    return TSNE(n_components=n, learning_rate='auto').fit_transform(data)

def proj_UMAP(data: np.ndarray, n: int) -> np.ndarray:
    return umap.UMAP(n_components=n).fit_transform(data)
##################################################################


############################ Models ##############################
def model_RF(data: np.ndarray, labels: np.ndarray) -> dict:
    """
    return a dictionary of test metrics:
    {
        'test_accuracy': [...]
        'test_precision': [...]
        'test_recall': [...]
        'test_f1': [...]
        'fit_time': [...]
        'score_time': [...]
    }
    For 5-fold cross-validation, each list has 5 entries
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42
    )
    scores = cross_validate(
        model, 
        data, 
        labels, 
        cv=5, 
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        return_train_score=False
    )
    
    return scores

def model_SVM(data: np.ndarray, labels: np.ndarray, kernel_type: str) -> dict:
    """
    Due to the unbalanced dataset, SVM struggles with classification
    We must explicitly state how to calculate the metrics in the 
    case that a test fold predicts only one class
    """
    precision = make_scorer(precision_score, zero_division=0)
    recall = make_scorer(recall_score, zero_division=0)
    f1 = make_scorer(f1_score, zero_division=0)



    K = np.dot(data, data.T)
    if kernel_type == "linear":
        model = SVC(kernel=kernel_type, random_state=42)
    elif kernel_type == "poly":
        degree = 3
        K = (K + 1)**degree
        model = SVC(kernel=kernel_type, degree=3, random_state=42)
    elif kernel_type == "rbf":
        # Normalize then exponentiate
        K = np.exp(K / np.max(np.abs(K)))
        model = SVC(kernel=kernel_type, random_state=42)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    scores = cross_validate(
        model, 
        data, 
        labels, 
        cv=5, 
        scoring={
            'accuracy': 'accuracy', 
            'precision': precision, 
            'recall': recall, 
            'f1': f1},
        return_train_score=False
    )

    return scores

def model_print_results(name: str, scores: dict):
    print(f"\n===== {name} (5-fold CV) =====")
    print(f"Accuracy:  {scores['test_accuracy'].mean():.4f}")
    print(f"Precision: {scores['test_precision'].mean():.4f}")
    print(f"Recall:    {scores['test_recall'].mean():.4f}")
    print(f"F1 Score:  {scores['test_f1'].mean():.4f}")
##################################################################
