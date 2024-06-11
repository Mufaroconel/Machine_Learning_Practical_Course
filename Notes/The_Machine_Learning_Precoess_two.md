### Data Pre-Processing

#### 1. Import the Data
Assuming you have already imported your data and displayed its head, let's move to the next step.

#### 2. Clean the Data
Ensure there are no missing values and the data types are correct.

```python
# Check for missing values
print(df.isnull().sum())

# Display data types
print(df.dtypes)
```

If there are no missing values and the data types are correct, we can proceed. Otherwise, we would handle missing values and correct data types accordingly.

#### 3. Split into Training & Test Sets
Split the dataset into training and testing sets.

```python
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('price_range', axis=1)
y = df['price_range']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 4. Feature Scaling
Standardize the features to have mean = 0 and variance = 1.

```python
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data, transform the test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Modelling

#### 5. Build the Model
We'll start with a Logistic Regression model, which is simple and effective for classification.

```python
from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression()
```

#### 6. Train the Model
Fit the model to the training data.

```python
# Train the model
model.fit(X_train_scaled, y_train)
```

#### 7. Make Predictions
Use the trained model to make predictions on the test data.

```python
# Make predictions
y_pred = model.predict(X_test_scaled)
```

### Evaluation

#### 8. Calculate Performance Metrics
Evaluate the model using accuracy and other relevant metrics.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
print(confusion_matrix(y_test, y_pred))
```

#### 9. Make a Verdict
Based on the performance metrics, we can decide if the model is good enough or if we need further improvements.

### Full Code Example
Here is the full code combined for clarity:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming df is already loaded

# Clean the Data
print(df.isnull().sum())
print(df.dtypes)

# Split into training & test sets
X = df.drop('price_range', axis=1)
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Model
model = LogisticRegression()

# Train the Model
model.fit(X_train_scaled, y_train)

# Make Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
```

This completes the guide for building, training, and evaluating your first machine learning model. If you encounter any specific issues or need further customization, feel free to ask!