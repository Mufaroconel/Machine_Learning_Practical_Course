### 1.5. Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.

SGD has been successfully applied to large-scale and sparse machine learning problems often encountered in text classification and natural language processing. Given that the data is sparse, the classifiers in this module easily scale to problems with more than \(10^5\) training examples and more than \(10^5\) features.

Strictly speaking, SGD is merely an optimization technique and does not correspond to a specific family of machine learning models. It is only a way to train a model. Often, an instance of `SGDClassifier` or `SGDRegressor` will have an equivalent estimator in the scikit-learn API, potentially using a different optimization technique. For example, using `SGDClassifier(loss='log_loss')` results in logistic regression, i.e. a model equivalent to `LogisticRegression` which is fitted via SGD instead of being fitted by one of the other solvers in `LogisticRegression`. Similarly, `SGDRegressor(loss='squared_error', penalty='l2')` and `Ridge` solve the same optimization problem, via different means.

The advantages of Stochastic Gradient Descent are:
- Efficiency.
- Ease of implementation (lots of opportunities for code tuning).

The disadvantages of Stochastic Gradient Descent include:
- SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations.
- SGD is sensitive to feature scaling.

#### Warning

Make sure you permute (shuffle) your training data before fitting the model or use `shuffle=True` to shuffle after each iteration (used by default). Also, ideally, features should be standardized using e.g. `make_pipeline(StandardScaler(), SGDClassifier())` (see Pipelines).

#### 1.5.1. Classification

The class `SGDClassifier` implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties for classification. Below is the decision boundary of a `SGDClassifier` trained with the hinge loss, equivalent to a linear SVM.

![SGD Separating Hyperplane](../_images/sphx_glr_plot_sgd_separating_hyperplane_001.png)

As other classifiers, SGD has to be fitted with two arrays: an array `X` of shape `(n_samples, n_features)` holding the training samples, and an array `y` of shape `(n_samples,)` holding the target values (class labels) for the training samples:

```python
from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X, y)
SGDClassifier(max_iter=5)
```

After being fitted, the model can then be used to predict new values:

```python
clf.predict([[2., 2.]])
array([1])
```

SGD fits a linear model to the training data. The `coef_` attribute holds the model parameters:

```python
clf.coef_
array([[9.9..., 9.9...]])
```

The `intercept_` attribute holds the intercept (aka offset or bias):

```python
clf.intercept_
array([-9.9...])
```

Whether or not the model should use an intercept, i.e. a biased hyperplane, is controlled by the parameter `fit_intercept`.

The signed distance to the hyperplane (computed as the dot product between the coefficients and the input sample, plus the intercept) is given by `SGDClassifier.decision_function`:

```python
clf.decision_function([[2., 2.]])
array([29.6...])
```

The concrete loss function can be set via the `loss` parameter. `SGDClassifier` supports the following loss functions:
- `loss="hinge"`: (soft-margin) linear Support Vector Machine,
- `loss="modified_huber"`: smoothed hinge loss,
- `loss="log_loss"`: logistic regression,

and all regression losses below. In this case, the target is encoded as -1 or 1, and the problem is treated as a regression problem. The predicted class then corresponds to the sign of the predicted target.

Please refer to the mathematical section below for formulas. The first two loss functions are lazy, they only update the model parameters if an example violates the margin constraint, which makes training very efficient and may result in sparser models (i.e. with more zero coefficients), even when L2 penalty is used.

Using `loss="log_loss"` or `loss="modified_huber"` enables the `predict_proba` method, which gives a vector of probability estimates per sample:

```python
clf = SGDClassifier(loss="log_loss", max_iter=5).fit(X, y)
clf.predict_proba([[1., 1.]])
array([[0.00..., 0.99...]])
```

The concrete penalty can be set via the `penalty` parameter. SGD supports the following penalties:
- `penalty="l2"`: L2 norm penalty on `coef_`.
- `penalty="l1"`: L1 norm penalty on `coef_`.
- `penalty="elasticnet"`: Convex combination of L2 and L1; (1 - `l1_ratio`) * L2 + `l1_ratio` * L1.

The default setting is `penalty="l2"`. The L1 penalty leads to sparse solutions, driving most coefficients to zero. The Elastic Net solves some deficiencies of the L1 penalty in the presence of highly correlated attributes. The parameter `l1_ratio` controls the convex combination of L1 and L2 penalty.

`SGDClassifier` supports multi-class classification by combining multiple binary classifiers in a “one versus all” (OVA) scheme. For each of the classes, a binary classifier is learned that discriminates between that and all other classes. At testing time, we compute the confidence score (i.e. the signed distances to the hyperplane) for each classifier and choose the class with the highest confidence. The Figure below illustrates the OVA approach on the iris dataset. The dashed lines represent the three OVA classifiers; the background colors show the decision surface induced by the three classifiers.

![SGD Iris](../_images/sphx_glr_plot_sgd_iris_001.png)

In the case of multi-class classification `coef_` is a two-dimensional array of shape `(n_classes, n_features)` and `intercept_` is a one-dimensional array of shape `(n_classes,)`. The i-th row of `coef_` holds the weight vector of the OVA classifier for the i-th class; classes are indexed in ascending order (see attribute `classes_`). Note that, in principle, since they allow to create a probability model, `loss="log_loss"` and `loss="modified_huber"` are more suitable for one-vs-all classification.

`SGDClassifier` supports both weighted classes and weighted instances via the fit parameters `class_weight` and `sample_weight`. See the examples below and the docstring of `SGDClassifier.fit` for further information.

`SGDClassifier` supports averaged SGD (ASGD). Averaging can be enabled by setting `average=True`. ASGD performs the same updates as the regular SGD, but instead of using the last value of the coefficients as the `coef_` attribute (i.e. the values of the last update), `coef_` is set instead to the average value of the coefficients across all updates. The same is done for the `intercept_` attribute. When using ASGD, the learning rate can be larger and even constant, leading on some datasets to a speed up in training time.

For classification with a logistic loss, another variant of SGD with an averaging strategy is available with Stochastic Average Gradient (SAG) algorithm, available as a solver in `LogisticRegression`.

**Examples**

- SGD: Maximum margin separating hyperplane
- Plot multi-class SGD on the iris dataset
- SGD: Weighted samples
- Comparing various online solvers
- SVM: Separating hyperplane for unbalanced classes (See the Note in the example)

#### 1.5.2. Regression

The class `SGDRegressor` implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties to fit linear regression models. `SGDRegressor` is well suited for regression problems with a large number of training samples (> 10,000), for other problems we recommend `Ridge`, `Lasso`, or `ElasticNet`.

The concrete loss function can be set via the `loss` parameter. `SGDRegressor` supports the following loss functions:
- `loss="squared_error"`: Ordinary least squares,
- `loss="huber"`: Huber loss for robust regression,
- `loss="epsilon_insensitive"`: linear Support Vector Regression.

Please refer to the mathematical section below for formulas. The Huber and epsilon-insensitive loss functions can be used for robust regression. The width of the insensitive region has to be specified via the parameter `epsilon`. This parameter depends on the scale of the target variables.

The `penalty` parameter determines the regularization to be used (see description above in the classification section).

`SGDRegressor` also supports averaged SGD (here again, see description above in the classification section).

For regression with a squared loss and a l2 penalty, another variant of SGD with an averaging strategy is available with Stochastic Average Gradient (

SAG) algorithm, available as a solver in `Ridge`.

**Examples**

- Robust linear model estimation using SGDRegressor
- Out-of-core classification of text documents

#### 1.5.3. Mathematical formulation

**Loss functions**

Let \(\mathbf{x}_i \in \mathbb{R}^p\) be the i-th training sample from a training set of size \(n\). Let \(y_i\) be the corresponding label (classification) or target (regression). For classification, we use \(\mathbf{x}_i\) for the features of the i-th sample and \(y_i\) for its label. The model then writes as:

\[
f(\mathbf{x}_i) = \mathbf{w}^\top \mathbf{x}_i + b
\]

where \(\mathbf{w}\) is a weight vector and \(b\) is the intercept. The regularized empirical risk minimization problem is:

\[
\min_{\mathbf{w}, b} \frac{1}{n} \sum_{i=1}^n \mathcal{L}(\mathbf{w}; \mathbf{x}_i, y_i) + \alpha R(\mathbf{w})
\]

where \( \mathcal{L} \) is a loss function, and \( R \) is a regularization term, multiplied by a constant \(\alpha\) (which we usually call the 'learning rate').

**Classification**

*Hinge loss*

For linear SVM, hinge loss is used:

\[
\mathcal{L}(\mathbf{w}; \mathbf{x}_i, y_i) = \max\{0, 1 - y_i (\mathbf{w}^\top \mathbf{x}_i + b)\}
\]

*Log loss*

For logistic regression, log loss is used:

\[
\mathcal{L}(\mathbf{w}; \mathbf{x}_i, y_i) = \log(1 + \exp(-y_i (\mathbf{w}^\top \mathbf{x}_i + b)))
\]

**Regression**

*Squared loss*

For linear regression, squared loss is used:

\[
\mathcal{L}(\mathbf{w}; \mathbf{x}_i, y_i) = \frac{1}{2} (\mathbf{w}^\top \mathbf{x}_i + b - y_i)^2
\]

*Huber loss*

For robust regression, Huber loss is used:

\[
\mathcal{L}(\mathbf{w}; \mathbf{x}_i, y_i) =
\begin{cases} 
\frac{1}{2} (\mathbf{w}^\top \mathbf{x}_i + b - y_i)^2 & \text{if} \, | \mathbf{w}^\top \mathbf{x}_i + b - y_i | \leq \epsilon \\
\epsilon | \mathbf{w}^\top \mathbf{x}_i + b - y_i | - \frac{1}{2} \epsilon^2 & \text{otherwise}
\end{cases}
\]

**Optimization**

SGD updates the model parameters by means of a single training example at a time. Given an example \((\mathbf{x}_i, y_i)\), the model parameters \(\mathbf{w}\) and \(b\) are updated as follows:

\[
\mathbf{w} \leftarrow \mathbf{w} - \eta \left( \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}; \mathbf{x}_i, y_i) + \alpha \nabla_{\mathbf{w}} R(\mathbf{w}) \right)
\]
\[
b \leftarrow b - \eta \nabla_b \mathcal{L}(\mathbf{w}; \mathbf{x}_i, y_i)
\]

where \(\eta\) is the learning rate. The learning rate can be either constant or gradually decaying. Different update rules are supported: constant (`learning_rate="constant"`), inv-scaling (`learning_rate="invscaling"`) and adaptive (`learning_rate="adaptive"`).
