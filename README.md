# State-and-Proj

This project provides a unified pipeline for:
- computing statistical features,
- performing dimenionality-reduction projections (PCA, FLD, t-SNE, UMAP)
- training classical and deep-learning models,
- and visualizing both raw data and classical kernel matrices.

It supports CSV datasets and EEG `.set` files, converting both into a consistent NumPy-based interface.

The pipeline extracts statistical descriptors, trains multiple machine-learning models, and generates projection plots for both the original dataset and the classical kernels derived from SVM (linear, polynomial, and RBF).

The goal is to offer a clean, reproducible demonstraction of feature extraction, model benchmarking, and manifold visualization across heterogeneous datasets.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Lrseward22/Stats-and-Proj
```

---

## Environment Setup

Create a virtual environment:

`python -m venv env`

Activate it:
- **Windows**: `env\Scripts\activate`

- **macOs/Linux**: `source env/bin/activate`

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the program:

```bash
python main.py
```

Inside `main.py`, toggle between the wine dataset and EEG data.

```python
USE_WINE = True       # set to False to use EEG data
```

---

## Dataset Requirement

Your dataset must follow these rules:

 - Rows = samples
 - Columns = features
 - All values must be numeric
 - EEG files must be `.set ` format

Wine labels are automatically generated:

 - white wine -> 0
 - red wine -> 1

EEG data is unlabeled, FLD is skipped automatically as there is not enough classes to facillitate it.

---

## Output

### Statistical Features

For each column, the program computes:
- mean
- standard deviation
- variance
- minimum/maximum
- interquartile range
- skewness
- kurtosis

These are printed in a structured format for inspection.

### Machine Learning Models

The following models are trained and evaluated using 5-fold cross-validation
- Random Forest (RF):
  - Decision trees are formed with decision thresholds optimized to suit a specified subset of the training data
- Support Vector Machine (SVM):
  - Delineats possible classes utilizing a hyperplane that maximizes the distance between the separate classes. To try to make sure the data is linear, we can try transformations according to the different kernels:
    - linear 
    - polynomial 
    - radial basis function (RBF)
- One-Class SVM:
  - Learns a boundary around a single class for anomaly‑style separation.
- Multi-Layer Perceptron (MLP)
  - Fully‑connected neural network for nonlinear classification.
- Convolutional Neural Network (CNN)
  - 1D‑CNN with two convolution layers followed by dense layers.

Each model reports:
- accuracy
- precision
- recall
- f1 score

### Projection Visualizations

Dimensionality-reduction plots are generated using Seaborn and saved to the `output/` directory.

- Principal Component Analysis (PCA):
    - Determines which dimensions the dataset has the largest variance and projects the data into only those dimensions. That is, it preserves only the dimensions of greatest variability. It preserves majority of variance and distances along principle directions.
- Fisher's Linear Discriminant (FLD):
    - Maximize class separability to ensure classes remain clustered. This preserves the class structure and decision boundaries between classes. It misses out on variance and nonlinear class boundaries.
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
    - Preserve local neighborhoods to create distinct clusters but distort global distances. Points that are close remain close however the distances between clusters lose meaning and the shape of the dataset is not preserved. Disregard cluster sizes.
- Uniform Manifold Approximation and Projection (UMAP)
    - Preserve local neighborhoods while maintaining manifold structure. It is faster and more stable.

 These projections are applied both the the *raw dataset* and to the *classical SVM kernels* (linear, polynomial, RBF), enabling visualization of how kernel geometry transforms the data.
 
The plots may be seen in the output folder [here](output/)

### Example Output

```
feature_0
  Mean:         6.2120
  STD:          9.1278
  Variance:     83.3159
  Min:          0.0000
  Max:          34.0000
  IQR:          7.2350
  Skew:         2.2506
  Kurtosis:     4.2375

.
.
.

===== MLP (5-fold CV) =====
Accuracy:  0.9823
Precision: 0.9827
Recall:    0.9453
F1 Score:  0.9635
```

---

## Project Structure

```
Stats-and-Proj/
│
├── main.py
├── utils.py
├── requirements.txt
└── dataset/
    ├── winewuality-red.csv
    ├── winequality-white.csv
    └── A_17.set
```

---

## Notes

- FLD produces a 1-D projection for 2-class datasets.
- EEG data is transposed so columns represents features.
- All processing is performed using NumPy arrays for consistency.
- kernel matrices are normalized before projection to stabilize t-SNE and UMAP


