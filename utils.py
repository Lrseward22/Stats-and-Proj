import numpy as np
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as FLD
from sklearn.manifold import TSNE
import umap


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
