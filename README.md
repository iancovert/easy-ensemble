# Easy-Ensemble

This package provides an easy-to-use tool to learn optimal model ensembles. Simply provide predictions from each of your models, and `easy-ensemble` will learn the optimal weights for combining your models.

## Installation

The easiest way to use this package is to clone the repository and install it in your Python environment:

```bash
git clone https://github.com/iancovert/easy-ensemble.git
cd easy-ensemble
pip install .
```

## Usage

The usage pattern follows scikit-learn semantics: all you need to do is set up the ensemble, fit it to your data using the `fit()` function, and then apply it using the `predict()` function.

A basic example looks like this:

```python
from ensemble import Ensemble

# Prepare data and models
X_train, Y_train = ...
X_val, Y_val = ...
X_test, Y_test = ...
models = ...

# Fit ensemble with mse objective, non-negative weights
ensemble = Ensemble('binary:logloss_logits', 'nonnegative')
preds = [model.predict(X_val) for model in models]
ensemble.fit(preds, Y_val)

# Apply ensemble
ensemble.predict([model.predict(X_test) for model in models])
```

For more detailed examples, please see the following notebooks:
- Binary classification using the adult dataset ([notebook](https://github.com/iancovert/easy-ensemble/blob/main/notebooks/adult.ipynb))
- Regression using the boston housing dataset ([notebook](https://github.com/iancovert/easy-ensemble/blob/main/notebooks/boston.ipynb))

## Description

After training multiple models for a machine learning task, the most common next steps are *model selection* (identifying the best single model) and *ensembling* (combining the models). The latter can perform better by letting the models correct each other's mistakes, particularly when the models are diverse or decorrelated (e.g., they're from different model classes). In the simplest version of model ensembling, the predictions are simply averaged, but it makes sense to weight the models differently when some are more accurate than others.

This package provides a simple way to learn the optimal weighting for your ensemble. And because such ensembles can overfit, you can apply constraints on the learned weights. The constraint options are:

- `'nonnegative'`: the weights cannot have negative values
- `'simplex'`: the weights must be in the probability simplex (they must be non-negative and sum to one)
- `'none'`: the weights are unconstrained (note that the weights may large when using correlated models)

Additionally, you can optimize your ensemble using a couple different loss functions. The options currently supported are:

- `'reg:mse'`: mean squared error loss
- `'binary:logloss_probs'`: log-loss for binary classification, with ensembling performed on the probabilities
- `'binary:logloss_logits'`: log-loss for binary classification, with ensembling performed on the logits
- `'multi:logloss_probs'`: log-loss for multi-class classification, with ensembling performed on the probabilities
- `'multi:logloss_logits'`: log-loss for multi-class classification, with ensembling performed on the logits

The objective function and constraints are the two main arguments required when setting up your ensemble. For example, you might set up your ensemble as follows:

```python
from ensemble import Ensemble
ensemble = Ensemble('reg:mse', 'simplex')
```

## How it works

When your loss function is convex (e.g., mean squared error or log-loss), finding the optimal ensemble weights requires solving a convex optimization problem. Adding constraints to the weights does not affect the convexity, but it does make finding the optimal solution a bit more difficult. Here, we find the optimal ensemble weights using *sequential quadratic programming* (SQP), which means that we solve a sequence of quadratic programs (QPs) that approximate the true objective around the current solution. To solve the underlying QPs, we use the excellent [osqp](https://github.com/osqp/osqp) package. While you could find the optimal ensemble weights using projected gradient descent, SQP is very fast and basically hyperparameter-free (no learning rate required).

## Contact

If you have any questions about the package, feel free to email me at <icovert@cs.washington.edu>.
