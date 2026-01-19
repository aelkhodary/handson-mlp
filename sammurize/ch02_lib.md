# Scikit-Learn Library Reference - Chapter 2

> All sklearn components used in the End-to-End Machine Learning Project

---

## 📐 Scikit-Learn Design Philosophy

Scikit-Learn's API is remarkably well designed. These are the main design principles:

- **Consistency** — All objects share a consistent and simple interface
- **Inspection** — Hyperparameters are public attributes; learned parameters have `_` suffix
- **Nonproliferation** — Uses NumPy arrays and SciPy matrices, not custom classes
- **Composition** — Build complex pipelines from simple building blocks
- **Sensible defaults** — Quick baseline with minimal configuration

---

## 🔧 Core Estimator Types

### Estimators
Any object that can estimate parameters from data. Uses `fit()` method.

```python
imputer = SimpleImputer(strategy="median")  # hyperparameter
imputer.fit(housing_num)  # learns from data
```

### Transformers
Estimators that can transform data. Uses `transform()` or `fit_transform()`.

```python
X = imputer.transform(housing_num)
# OR (does both at once, often optimized)
X = imputer.fit_transform(housing_num)
```

### Predictors
Estimators that make predictions. Uses `predict()` and `score()`.

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
```

---

# 📊 Data Splitting & Validation

## `train_test_split`
**Module:** `sklearn.model_selection`

**Purpose:** Split data into training and test sets randomly.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # reproducibility
)
```

**When to use:** Quick and simple splits for initial experiments.

---

## `StratifiedShuffleSplit`
**Module:** `sklearn.model_selection`

**Purpose:** Split data while maintaining the same proportion of samples for each class/category.

```python
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

**When to use:** When you have important categorical features that must be represented proportionally in both sets (e.g., income categories).

---

## `cross_val_score`
**Module:** `sklearn.model_selection`

**Purpose:** Evaluate model using k-fold cross-validation.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y,
    scoring="neg_root_mean_squared_error",
    cv=10  # 10 folds
)
rmse_scores = -scores  # flip sign (sklearn uses "higher is better")
print(f"Mean: {rmse_scores.mean():.0f}, Std: {rmse_scores.std():.0f}")
```

**What you get:**
- Mean score → Expected performance on new data
- Standard deviation → How reliable the estimate is

**When to use:** Always! More reliable than a single validation split.

---

## `GridSearchCV`
**Module:** `sklearn.model_selection`

**Purpose:** Exhaustively search all combinations of hyperparameters.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': [4, 6, 8]
}

grid_search = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    return_train_score=True
)
grid_search.fit(X, y)

print(grid_search.best_params_)       # {'max_features': 6, 'n_estimators': 200}
print(grid_search.best_estimator_)    # The best model, already fitted
```

**When to use:** Small hyperparameter spaces where you can afford to try every combination.

---

## `RandomizedSearchCV`
**Module:** `sklearn.model_selection`

**Purpose:** Random search over hyperparameter distributions.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_features': randint(2, 10),
    'max_depth': [None, 10, 20, 30]
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(),
    param_distributions,
    n_iter=20,  # try 20 random combinations
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42
)
random_search.fit(X, y)
```

**Advantages over GridSearchCV:**
- Explores more values for continuous hyperparameters
- Fixed iterations regardless of search space size
- Often finds good solutions faster

**When to use:** Large hyperparameter spaces, continuous hyperparameters.

---

## `HalvingRandomSearchCV`
**Module:** `sklearn.model_selection` (experimental)

**Purpose:** Progressive elimination search - starts with many candidates on small data, progressively increases data for best performers.

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

halving_search = HalvingRandomSearchCV(
    RandomForestRegressor(),
    param_distributions,
    factor=2,  # double resources each round
    random_state=42
)
```

**When to use:** Very large search spaces where full CV on all candidates is too slow.

---

# 🧹 Data Preprocessing

## `SimpleImputer`
**Module:** `sklearn.impute`

**Purpose:** Replace missing values with a computed statistic.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")  # or "mean", "most_frequent", "constant"
imputer.fit(housing_num)

# Learned values stored in:
print(imputer.statistics_)  # median of each feature

X_filled = imputer.transform(housing_num)
```

**Strategies:**
| Strategy | Use Case |
|----------|----------|
| `"median"` | Numerical data with outliers (most common) |
| `"mean"` | Numerical data without outliers |
| `"most_frequent"` | Categorical data |
| `"constant"` | Fill with specific value (`fill_value=...`) |

**Why use it over pandas?**
- Remembers statistics from training data
- Applies same imputation to new data in production

---

## `OrdinalEncoder`
**Module:** `sklearn.preprocessing`

**Purpose:** Convert categorical values to ordered integers.

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
housing_cat_encoded = encoder.fit_transform(housing[["ocean_proximity"]])

# Categories learned:
print(encoder.categories_)
# [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])]
```

**Result:**
```
'<1H OCEAN' → 0
'INLAND'    → 1
'ISLAND'    → 2
'NEAR BAY'  → 3
'NEAR OCEAN'→ 4
```

**⚠️ Warning:** Only use for naturally ordered categories (e.g., "low", "medium", "high"). ML algorithms will assume 4 > 3 > 2 > 1.

---

## `OneHotEncoder`
**Module:** `sklearn.preprocessing`

**Purpose:** Convert categorical values to binary columns (one column per category).

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)  # dense array output
housing_cat_1hot = encoder.fit_transform(housing[["ocean_proximity"]])

print(encoder.categories_)
print(encoder.feature_names_in_)
```

**Result for "NEAR OCEAN":**
```
[0, 0, 0, 0, 1]  # Only the 5th column is 1
```

**Why use over `pd.get_dummies()`?**
- Remembers categories from training
- Handles unknown categories gracefully (`handle_unknown="ignore"`)
- Ensures consistent features in production

---

## `MinMaxScaler`
**Module:** `sklearn.preprocessing`

**Purpose:** Scale features to a specific range (default [0, 1]).

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))  # can change range
X_scaled = scaler.fit_transform(X)

# Learned parameters:
print(scaler.data_min_)   # min of each feature
print(scaler.data_max_)   # max of each feature
```

**Formula:** `X_scaled = (X - X_min) / (X_max - X_min)`

**⚠️ Sensitive to outliers:** One extreme value can crush all others.

---

## `StandardScaler`
**Module:** `sklearn.preprocessing`

**Purpose:** Standardize features to zero mean and unit variance.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Learned parameters:
print(scaler.mean_)   # mean of each feature
print(scaler.scale_)  # std dev of each feature
```

**Formula:** `X_scaled = (X - mean) / std`

**✅ Preferred for most cases:** More robust to outliers than MinMaxScaler.

---

## `FunctionTransformer`
**Module:** `sklearn.preprocessing`

**Purpose:** Apply any custom function as a transformer.

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_population = log_transformer.fit_transform(housing[["population"]])
```

**Use case:** Apply log transform, square root, or any custom function within a pipeline.

---

# 🤖 Models (Estimators)

## `LinearRegression`
**Module:** `sklearn.linear_model`

**Purpose:** Fit a linear model that minimizes squared errors.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(model.coef_)       # feature weights
print(model.intercept_)  # bias term
```

**Characteristics:**
- Simple, fast, interpretable
- Often underfits complex data
- Good baseline model to start with

---

## `DecisionTreeRegressor`
**Module:** `sklearn.tree`

**Purpose:** Build a tree that splits data based on feature thresholds.

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(model.feature_importances_)  # importance of each feature
```

**Characteristics:**
- Can capture non-linear relationships
- **Prone to overfitting** (0 training error is a red flag!)
- Use as a building block for ensembles

---

## `RandomForestRegressor`
**Module:** `sklearn.ensemble`

**Purpose:** Ensemble of decision trees that averages their predictions.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,    # number of trees
    max_features="sqrt", # features per split
    n_jobs=-1,           # use all CPU cores
    random_state=42
)
model.fit(X_train, y_train)

print(model.feature_importances_)
```

**Characteristics:**
- **Best performer in Chapter 2**
- Reduces overfitting vs. single tree
- Provides feature importance scores
- Parallelizable (use `n_jobs=-1`)

---

## `SVR` (Support Vector Regressor)
**Module:** `sklearn.svm`

**Purpose:** Regression using support vector machines.

```python
from sklearn.svm import SVR

model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
model.fit(X_train, y_train)
```

**Kernels:**
- `"linear"` - Linear relationships
- `"rbf"` - Non-linear (Gaussian)
- `"poly"` - Polynomial

**When to use:** Small to medium datasets, especially when you need non-linear modeling.

---

## `KNeighborsRegressor`
**Module:** `sklearn.neighbors`

**Purpose:** Predict based on average of k nearest neighbors.

```python
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**When to use:** Simple baseline, spatial data, or as a feature transformer.

---

## `KMeans`
**Module:** `sklearn.cluster`

**Purpose:** Cluster data into k groups.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Cluster centers:
print(kmeans.cluster_centers_)
```

**Use in Chapter 2:** Create cluster-based features (e.g., similarity to cluster centers).

---

## `IsolationForest`
**Module:** `sklearn.ensemble`

**Purpose:** Detect outliers/anomalies in data.

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_pred = iso_forest.fit_predict(X)
# -1 = outlier, 1 = normal
```

**When to use:** Identify and optionally remove outliers before training.

---

# 🔗 Pipelines & Composition

## `Pipeline`
**Module:** `sklearn.pipeline`

**Purpose:** Chain multiple transformers and a final estimator.

```python
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_prepared = num_pipeline.fit_transform(X)
```

**Benefits:**
- Clean, readable code
- Prevents data leakage (fit only on training data)
- Easy to save and deploy

---

## `make_pipeline`
**Module:** `sklearn.pipeline`

**Purpose:** Create pipeline with auto-generated step names.

```python
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)
# Names: "simpleimputer", "standardscaler"
```

---

## `ColumnTransformer`
**Module:** `sklearn.compose`

**Purpose:** Apply different transformations to different columns.

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),      # numerical columns
    ("cat", OneHotEncoder(), cat_cols)    # categorical columns
])

X_prepared = preprocessor.fit_transform(X)
```

---

## `make_column_transformer` & `make_column_selector`
**Module:** `sklearn.compose`

**Purpose:** Quick column transformer with auto-selection by dtype.

```python
from sklearn.compose import make_column_selector, make_column_transformer

preprocessor = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(), make_column_selector(dtype_include=object))
)
```

---

## `TransformedTargetRegressor`
**Module:** `sklearn.compose`

**Purpose:** Transform target variable (y) during training.

```python
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log,          # transform y before fit
    inverse_func=np.exp   # inverse for predictions
)
```

**When to use:** When target has skewed distribution (e.g., log-transform prices).

---

# 📏 Metrics

## `root_mean_squared_error`
**Module:** `sklearn.metrics`

**Purpose:** Calculate RMSE between predictions and true values.

```python
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_true, y_pred)
print(f"RMSE: ${rmse:,.0f}")
```

---

## `rbf_kernel`
**Module:** `sklearn.metrics.pairwise`

**Purpose:** Compute similarity between samples using Gaussian RBF kernel.

```python
from sklearn.metrics.pairwise import rbf_kernel

similarities = rbf_kernel(X, cluster_centers, gamma=0.1)
```

**Use in Chapter 2:** Create features based on similarity to cluster centers.

---

# 🎯 Feature Selection

## `SelectFromModel`
**Module:** `sklearn.feature_selection`

**Purpose:** Automatically select features based on model importance.

```python
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(
    RandomForestRegressor(random_state=42),
    threshold="median"  # keep features above median importance
)
selector.fit(X, y)
X_reduced = selector.transform(X)

# Selected features:
print(selector.get_support())
```

---

# 🏗️ Base Classes (For Custom Transformers)

## `BaseEstimator` & `TransformerMixin`
**Module:** `sklearn.base`

**Purpose:** Build custom transformers compatible with sklearn API.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomRatioTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_extra_feature=True):
        self.add_extra_feature = add_extra_feature
    
    def fit(self, X, y=None):
        return self  # nothing to learn
    
    def transform(self, X):
        rooms_per_house = X[:, 0] / X[:, 1]  # total_rooms / households
        if self.add_extra_feature:
            return np.c_[X, rooms_per_house]
        return rooms_per_house.reshape(-1, 1)
```

**What you get from mixins:**
- `TransformerMixin` → `fit_transform()` method
- `BaseEstimator` → `get_params()` and `set_params()` for GridSearch

---

## `clone`
**Module:** `sklearn.base`

**Purpose:** Create a new unfitted copy of an estimator.

```python
from sklearn.base import clone

new_model = clone(original_model)  # same hyperparams, not fitted
```

---

# ✅ Validation Utilities

## `check_is_fitted`
**Module:** `sklearn.utils.validation`

**Purpose:** Check if an estimator has been fitted.

```python
from sklearn.utils.validation import check_is_fitted

check_is_fitted(model)  # raises error if not fitted
```

---

## `check_estimator`
**Module:** `sklearn.utils.estimator_checks`

**Purpose:** Verify custom estimator follows sklearn API.

```python
from sklearn.utils.estimator_checks import check_estimator

check_estimator(CustomTransformer())  # runs all API tests
```

---

# 📋 Quick Reference Table

| Module | Class | Purpose |
|--------|-------|---------|
| `model_selection` | `train_test_split` | Split data |
| `model_selection` | `StratifiedShuffleSplit` | Stratified splits |
| `model_selection` | `cross_val_score` | Cross-validation |
| `model_selection` | `GridSearchCV` | Exhaustive hyperparameter search |
| `model_selection` | `RandomizedSearchCV` | Random hyperparameter search |
| `impute` | `SimpleImputer` | Fill missing values |
| `preprocessing` | `OrdinalEncoder` | Categories → integers |
| `preprocessing` | `OneHotEncoder` | Categories → binary columns |
| `preprocessing` | `MinMaxScaler` | Scale to [0,1] |
| `preprocessing` | `StandardScaler` | Standardize (z-score) |
| `preprocessing` | `FunctionTransformer` | Custom functions |
| `linear_model` | `LinearRegression` | Linear model |
| `tree` | `DecisionTreeRegressor` | Decision tree |
| `ensemble` | `RandomForestRegressor` | Random forest |
| `ensemble` | `IsolationForest` | Outlier detection |
| `svm` | `SVR` | Support vector regression |
| `neighbors` | `KNeighborsRegressor` | K-nearest neighbors |
| `cluster` | `KMeans` | K-means clustering |
| `pipeline` | `Pipeline` | Chain transformers |
| `compose` | `ColumnTransformer` | Different transforms per column |
| `compose` | `TransformedTargetRegressor` | Transform target |
| `metrics` | `root_mean_squared_error` | Calculate RMSE |
| `feature_selection` | `SelectFromModel` | Auto feature selection |
| `base` | `BaseEstimator` | Custom estimator base |
| `base` | `TransformerMixin` | Custom transformer mixin |

---

# 🧠 Key Takeaways

1. **Start with `SimpleImputer` + `StandardScaler`** — handles 90% of preprocessing needs

2. **Always use `Pipeline`** — prevents data leakage and makes code cleaner

3. **Use `cross_val_score` instead of single splits** — more reliable evaluation

4. **`RandomizedSearchCV` > `GridSearchCV`** for large search spaces

5. **Custom transformers are easy** — just inherit from `BaseEstimator` + `TransformerMixin`

6. **Check `feature_importances_`** — understand what your model learned
