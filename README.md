# State-and-Proj

A simple demonstration project for computing statistical features and running dimensionality-reduction projections(PCA, FLD, t-SNE, UMAP) on arbitrary datasets. Supports both CSV datasets and EEG .set files through a unified NumPy interface.

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

The program prints:

- mean
- standard deviation
- variance
- min/max
- interquartile range
- skewness
- kurtosis

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
```

And generates projections plots using Seaborn:

- Principal Component Analysis (PCA):
    - Determines which dimensions the dataset has the largest variance and projects the data into only those dimensions. That is, it preserves only the dimensions of greatest variability. It preserves majority of variance and distances along principle directions.
- Fisher's Linear Discriminant (FLD):
    - Maximize class separability to ensure classes remain clustered. This preserves the class structure and decision boundaries between classes. It misses out on variance and nonlinear class boundaries.
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
    - Preserve local neighborhoods to create distinct clusters but distort global distances. Points that are close remain close however the distances between clusters lose meaning and the shape of the dataset is not preserved. Disregard cluster sizes.
- Uniform Manifold Approximation and Projection (UMAP)
    - Preserve local neighborhoods while maintaining manifold structure. It is faster and more stable.
 
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
- All processing is down using NumPy arrays for consistency.



