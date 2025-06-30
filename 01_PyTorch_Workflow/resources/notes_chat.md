### **Universal Splitting Principles (Regression and Classification)**

Data splitting is a critical step in machine learning to ensure models generalize well to unseen data. The following principles apply to both regression and classification tasks:

1. **Standard Split Ratios**:
   - **Train/Validation/Test Split**: A common practice is to split data into three subsets:
     - **Training set**: Used to train the model (typically 60-80% of data).
     - **Validation set**: Used for hyperparameter tuning and model selection (10-20% of data).
     - **Test set**: Used for final evaluation of the model's performance (10-20% of data).
   - Common ratios: **80/10/10**, **70/15/15**, or **60/20/20**.

2. **Random vs. Stratified Splitting**:
   - **Random Splitting**: Randomly assign data points to train/validation/test sets. Suitable for regression and balanced classification datasets.
   - **Stratified Splitting**: Ensures the same class distribution in each split as in the original dataset. Essential for imbalanced classification datasets to maintain representativeness.

3. **Avoid Data Leakage**:
   - Ensure no data from the test or validation set is used during training (e.g., feature scaling should be fit only on the training set).
   - Temporal data requires time-based splitting to prevent future information leakage.

4. **Cross-Validation**:
   - Use **k-fold cross-validation** (e.g., 5-fold or 10-fold) for robust evaluation, especially with small datasets.
   - For classification, use **stratified k-fold** to maintain class distribution in each fold.

5. **Reproducibility**:
   - Set random seeds for reproducibility in splitting and model training.

6. **Dataset Size Considerations**:
   - **Large datasets**: Simple train/validation/test splits are sufficient.
   - **Small datasets**: Use cross-validation to maximize training data usage and reduce variance in performance estimates.

7. **Domain-Specific Splitting**:
   - For time-series data, use **time-based splitting** (e.g., earlier data for training, later data for testing).
   - For grouped data (e.g., patients in medical datasets), ensure all samples from a group are in the same split to avoid leakage.

---

### **Table of Concepts and Their Meanings**

| **Concept**                | **Meaning**                                                                 | **When to Use**                                                                 |
|----------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **Train/Validation/Test Split** | Divides data into three sets: training (model learning), validation (tuning), test (evaluation). | Standard practice for most ML tasks.                                           |
| **Random Splitting**        | Randomly assigns data points to splits without considering class distribution. | Regression or balanced classification datasets.                                |
| **Stratified Splitting**    | Ensures class distribution in each split matches the original dataset.         | Imbalanced classification datasets.                                            |
| **K-Fold Cross-Validation** | Splits data into k folds, trains on k-1 folds, validates on the remaining fold, repeating k times. | Small datasets or when robust performance estimation is needed.                |
| **Stratified K-Fold**       | K-fold cross-validation with class distribution preserved in each fold.        | Imbalanced classification datasets.                                            |
| **Time-Based Splitting**    | Splits data chronologically (earlier data for training, later for testing).    | Time-series data to avoid future leakage.                                      |
| **Group-Based Splitting**   | Ensures all samples from a group (e.g., patient) stay in the same split.      | Grouped data (e.g., medical, multi-session data) to avoid leakage.             |

---

### **Considerations and Constraints**

1. **Dataset Size**:
   - **Small datasets**: Use cross-validation to avoid overfitting and maximize training data.
   - **Large datasets**: A single train/validation/test split is often sufficient, but ensure enough data in validation/test sets for reliable evaluation.

2. **Class Imbalance** (Classification):
   - Use stratified splitting to ensure proportional representation of minority classes.
   - Consider oversampling (e.g., SMOTE) or undersampling only on the training set to avoid leakage.

3. **Temporal Data**:
   - Use time-based splits to mimic real-world scenarios where future data is unavailable during training.
   - Avoid random shuffling to prevent leakage of future information.

4. **Feature Preprocessing**:
   - Fit preprocessing steps (e.g., scaling, normalization) on the training set only and apply to validation/test sets to avoid data leakage.

5. **Reproducibility**:
   - Set random seeds for splitting and model initialization to ensure consistent results.

---

### **Warnings**

1. **Data Leakage**:
   - Using test/validation data during training (e.g., for feature selection or scaling) leads to overly optimistic performance estimates.
   - Example: Normalizing the entire dataset before splitting leaks test set information into the training process.

2. **Overfitting to Validation Set**:
   - Repeatedly tuning hyperparameters on the same validation set can lead to overfitting to that set. Use cross-validation or a separate test set for final evaluation.

3. **Ignoring Class Imbalance** (Classification):
   - Random splitting in imbalanced datasets can result in splits with no or few samples of minority classes, leading to biased models.

4. **Small Test Sets**:
   - A test set that is too small may not provide a reliable estimate of model performance. Ensure at least 10-20% of data for testing, or use cross-validation.

5. **Ignoring Domain Knowledge**:
   - Failing to account for domain-specific constraints (e.g., time-series or grouped data) can lead to invalid splits and poor generalization.

---

### **Best Practice Coding in PyTorch**

Below are examples of best practices for data splitting in PyTorch for both regression and classification, including random and stratified splits, cross-validation, and time-based splitting.

#### **1. Random Train/Validation/Test Split (Regression or Balanced Classification)**

```python
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)

# Example data (X: features, y: targets)
X = torch.randn(1000, 10)  # 1000 samples, 10 features
y = torch.randn(1000)      # Regression targets (or classification labels)

# Split into train (70%), validation (15%), test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print split sizes
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
```

**Notes**:
- `shuffle=True` in `train_loader` improves training by randomizing batch order.
- `shuffle=False` in `val_loader` and `test_loader` ensures consistent evaluation.

#### **2. Stratified Splitting (Classification)**

```python
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Set random seed
torch.manual_seed(42)

# Example classification data (imbalanced)
X = torch.randn(1000, 10)
y = torch.randint(0, 3, (1000,))  # 3 classes, imbalanced

# Ensure integer labels for stratification
y = y.numpy()
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Convert back to tensors
y_train = torch.tensor(y_train)
y_val = torch.tensor(y_val)
y_test = torch.tensor(y_test)

# Create datasets and loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Verify class distribution
print("Class distribution in train:", torch.bincount(y_train))
print("Class distribution in val:", torch.bincount(y_val))
print("Class distribution in test:", torch.bincount(y_test))
```

**Notes**:
- `stratify=y` ensures class proportions are maintained in each split.
- Check class distribution to confirm stratification worked.

#### **3. K-Fold Cross-Validation (Regression or Classification)**

```python
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

# Set random seed
torch.manual_seed(42)

# Example data
X = torch.randn(1000, 10)
y = torch.randint(0, 3, (1000,))  # Classification (use torch.randn for regression)

# K-Fold Cross-Validation (use StratifiedKFold for classification)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Use KFold for regression
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}")

    # Split data
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train and evaluate model here (placeholder)
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
```

**Notes**:
- Use `StratifiedKFold` for classification to preserve class distribution.
- Use `KFold` for regression or balanced classification.
- Average performance metrics across folds for robust evaluation.

#### **4. Time-Based Splitting (Time-Series Data)**

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Set random seed
torch.manual_seed(42)

# Example time-series data (1000 timesteps)
X = torch.randn(1000, 10)  # Features
y = torch.randn(1000)      # Targets

# Time-based split (e.g., 70% train, 15% val, 15% test)
n = len(X)
train_idx = int(0.7 * n)
val_idx = int(0.85 * n)

X_train, y_train = X[:train_idx], y[:train_idx]
X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
X_test, y_test = X[val_idx:], y[val_idx:]

# Create datasets and loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # No shuffle for time-series
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
```

**Notes**:
- `shuffle=False` to preserve temporal order.
- Ensure no future data leaks into training (e.g., no random splitting).

---

### **Best Practices Summary**

1. **Choose the Right Split Strategy**:
   - Random split for regression or balanced classification.
   - Stratified split for imbalanced classification.
   - Time-based split for time-series data.
   - Group-based split for grouped data.

2. **Prevent Data Leakage**:
   - Fit preprocessing (e.g., scaling) on training data only.
   - Avoid using test/validation data for feature engineering.

3. **Use Cross-Validation for Small Datasets**:
   - K-fold for regression, stratified k-fold for classification.

4. **Ensure Reproducibility**:
   - Set random seeds for splitting and model training.

5. **Validate Split Quality**:
   - Check class distribution in classification tasks.
   - Verify temporal order in time-series data.

6. **Optimize DataLoader Usage**:
   - Use `shuffle=True` for training to improve convergence.
   - Use `shuffle=False` for validation/test to ensure consistent evaluation.

---

### **Additional Notes**

- **PyTorch Datasets and DataLoaders**: Always use `TensorDataset` and `DataLoader` for efficient batching and shuffling in PyTorch.
- **Custom Datasets**: For complex datasets (e.g., images, text), create a custom `torch.utils.data.Dataset` class to handle loading and preprocessing.
- **Imbalanced Datasets**: Consider techniques like oversampling (SMOTE) or class weighting in the loss function, but apply them only to the training set.
- **Hardware Considerations**: Adjust `batch_size` based on GPU memory (e.g., 32, 64, or 128 are common).

