import pandas as pd
import numpy as np
import mne
import seaborn as sns
import matplotlib.pyplot as plt
import os

import utils

######################### Data Loading ###########################
# Load an arbitrary file (wine vs eeg) and ensure it is in a common interface
"""
We want to find the statistical features that describe each sample
EEG data has each column denoting the samples while the wine dataset 
has each row as a separate sample. Thus, we need to transpose the EEG data
"""
def load_file(path: str) -> np.ndarray:
    # Identifies it as an EEG signal
    if path.endswith(".set"):
        return load_set(path).T

    # Identifies it as a wine file
    elif path.endswith(".csv"):
        return load_csv(path).to_numpy()

    else:
        raise ValueError("Unsupported file type.")

def load_set(path:str) -> np.ndarray:
    raw = mne.io.read_raw_eeglab(path, preload=True)
    data = raw.get_data()
    return data

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path).apply(pd.to_numeric, errors='coerce')
##################################################################


########################### Get Stats ############################
"""
Initially, I thought I was getting statistical data per feature
rather than per sample. In the context of ML, it makes more sense 
to get them per sample and append them to each sample. As a result,
I have determined that quantiles lose much of there meaning for the
wine dataset. I have omitted them.
"""
def stats_extract(data: np.ndarray) -> dict:
    stats_dict = {}

    # Iterate per sample
    for i in range(data.shape[0]):
        x = data[i, :]

        stats_dict[f'sample_{i}'] = {
            "mean": utils.stats_mean(x),
            "std": utils.stats_SD(x),
            "var": utils.stats_variance(x),
            "min": utils.stats_min(x),
            "max": utils.stats_max(x),
            "iqr": utils.stats_interquartile_range(x),
            #"quant": utils.stats_quantiles(x, [25, 50, 75]),
            "skew": utils.stats_skew(x),
            "kurt": utils.stats_kurtosis(x)
        }

    return stats_dict

def stats_print(stats: dict, limit: int = 16):
    print("\n===== STATISTICS SUMMARY (showing first", limit, "samples) =====\n")

    count = 0

    for sample, values in stats.items():
        if count >= limit:
            break
        count += 1

        print(f"{sample}")
        print(f"  Mean:         {values['mean']:.4f}")
        print(f"  STD:          {values['std']:.4f}")
        print(f"  Variance:     {values['var']:.4f}")
        print(f"  Min:          {values['min']:.4f}")
        print(f"  Max:          {values['max']:.4f}")
        print(f"  IQR:          {values['iqr']:.4f}")
        print(f"  Skew:         {values['skew']:.4f}")
        print(f"  Kurtosis:     {values['kurt']:.4f}")
        #print(f"  Quantiles:    {values['quant']}")
        print()

def stats_to_numpy(stats: dict) -> np.ndarray:
    stats_arr = []
    for sample, values in stats.items():
        stats_arr.append([
            values['mean'],
            values['std'],
            values['var'],
            values['min'],
            values['max'],
            values['iqr'],
            values['skew'],
            values['kurt']
        ])
    return np.array(stats_arr, dtype=float)
##################################################################


########################## Projections ###########################
def proj_run(data: np.ndarray, labels: np.ndarray | None, dim: int, use_FLD=False) -> dict:
    projections = {}

    # PCA
    projections["PCA"] = utils.proj_PCA(data, dim)

    # FLD
    if use_FLD:
        projections["FLD"] = utils.proj_FLD(data, labels, dim)
    else:
        projections["FLD"] = None

    # TSNE
    projections["TSNE"] = utils.proj_TSNE(data, dim)

    # UMAP
    projections["UMAP"] = utils.proj_UMAP(data, dim)

    return projections

def proj_exist(projs: list[str], name: str, folder: str="output") -> bool:
    for proj in projs:
        filename = f"{name}_{proj}.png"
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            return False
    return True

def proj_plot(name: str, proj: np.ndarray, labels: np.ndarray | None, proj_name: str):
    if proj is None:
        return

    dims = proj.shape[1]

    plt.figure(figsize=(7,6))

    if dims == 1:
        if labels is None:
            sns.scatterplot(
                x=proj[:, 0],
                y=np.zeros_like(proj[:,0]),
                s=20,
                alpha=0.7,
            )
        else:
            sns.scatterplot(
                x=proj[:, 0],
                y=np.zeros_like(proj[:,0]),
                hue=labels,
                palette="coolwarm",
                s=20,
                alpha=0.8,
                legend=True
            )
            plt.xlabel("Component 1")
            plt.ylabel("")
            plt.title(f"{name} Projection (1D)")
            plt.yticks([])

    elif dims >= 2:
        if labels is None:
            sns.scatterplot(
                x=proj[:, 0],
                y=proj[:,1],
                s=20,
                alpha=0.7,
                edgecolor=None,
            )
        else:
            sns.scatterplot(
                x=proj[:, 0],
                y=proj[:,1],
                hue=labels,
                palette="coolwarm",
                s=20,
                alpha=0.8,
                edgecolor=None,
                legend=True
            )
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"{name} {proj_name} Projection")

    plt.tight_layout()
    plt.savefig(f"output/{name}_{proj_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

##################################################################


def load_wine() -> tuple[np.ndarray, np.ndarray]:
    # Load Wine dataset
    red = load_file("dataset/winequality-red.csv")
    white = load_file("dataset/winequality-white.csv")

    # Add labels
    red_labels = np.ones(red.shape[0], dtype=int)
    white_labels = np.zeros(white.shape[0], dtype=int)

    # Combine
    data = np.vstack([red, white])
    labels = np.concatenate([red_labels, white_labels])

    return data, labels

def main():
    USE_WINE = True
    USE_FLD = USE_WINE
    TRAIN_MODEL = USE_WINE

    if USE_WINE:
        data, labels = load_wine()
    else:
        data = load_file("dataset/A_17.set")
        labels = None

    stats = stats_extract(data)
    stats_print(stats)

    X_stats = stats_to_numpy(stats)
    X = np.concatenate([data, X_stats], axis=1)

    kernels = {}

    if TRAIN_MODEL:
        scores_RF = utils.model_RF(X, labels)
        utils.model_print_results("Random Forest", scores_RF)

        scores_SVM_linear, K_linear = utils.model_SVM(X, labels, "linear")
        utils.model_print_results("SVM - Linear", scores_SVM_linear)
        kernels["linear"] = K_linear

        scores_SVM_poly, K_poly = utils.model_SVM(X, labels, "poly")
        utils.model_print_results("SVM - Polynomial", scores_SVM_poly)
        kernels["poly"] = K_poly

        scores_SVM_rbf, K_rbf = utils.model_SVM(X, labels, "rbf")
        utils.model_print_results("SVM - RBF", scores_SVM_rbf)
        kernels["rbf"] = K_rbf

        scores_OneClassSVM = utils.model_OneClassSVM(X, labels, "linear")
        utils.model_print_results("One-Class SVM", scores_OneClassSVM)

        scores_MLP = utils.model_MLP(data, labels)
        utils.model_print_results("MLP", scores_MLP)

        scores_CNN = utils.model_CNN(data, labels)
        utils.model_print_results("CNN", scores_CNN)

    def plot_all_projections(projections: dict, labels: np.ndarray | None, data_name: str):
        for name, proj in projections.items():
            proj_plot(data_name, proj, labels, name)

    names = ["PCA", "FLD", "TSNE", "UMAP"] if USE_FLD else ["PCA", "TSNE", "UMAP"]
    if not proj_exist(names, "dataset"):
        projections = proj_run(data, labels, dim=2, use_FLD=USE_FLD)
        plot_all_projections(projections, labels, "dataset")

    for name, kernel in kernels.items():
        # Normalize to help with projections
        kernel = kernel / np.max(np.abs(kernel))
        if not proj_exist(names, name):
            projections = proj_run(kernel, labels, dim=2, use_FLD=USE_FLD)
            plot_all_projections(projections, labels, name)

if __name__ == "__main__":
    main()
