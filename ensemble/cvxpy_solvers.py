# TODO this file is not currently used

import numpy as np
import cvxpy as cp


def solve_regressor_mse(preds,
                        targets,
                        constraints='simplex',
                        verbose=False):
    '''
    Solve for optimal regressor ensemble using MSE objective function.
    
    Args:
      preds: list or tuple of each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights ('simplex',
        'nonnegative' or 'none').
      verbose: whether CVXPY solver should generate verbose output.

    Returns: weights for optimal ensemble.
    '''
    # Setup
    assert constraints in ['simplex', 'nonnegative', 'none']
    m = len(preds)
    
    # Objective
    w = cp.Variable(m)
    preds_stack = np.array(preds)
    ensemble_preds = w @ preds_stack
    objective = cp.sum_squares(ensemble_preds - targets)
    
    # Apply constraints
    if constraints == 'simplex':
        constraints = [w >= 0, cp.sum(w) == 1]
    elif constraints == 'nonnegative':
        constraints = [w >= 0]
    elif constraints == 'none':
        constraints = []
    
    # Solve problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver='ECOS', verbose=verbose)
    return w.value


def solve_regressor_mae(preds,
                        targets,
                        constraints='simplex',
                        verbose=False):
    '''
    Solve for optimal regressor ensemble using MAE objective function.
    
    Args:
      preds: list or tuple of each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights ('simplex',
        'nonnegative' or 'none').
      verbose: whether CVXPY solver should generate verbose output.

    Returns: weights for optimal ensemble.
    '''
    # Setup
    assert constraints in ['simplex', 'nonnegative', 'none']
    n = len(preds[0])
    m = len(preds)
    
    # Objective
    w = cp.Variable(m)
    preds_stack = np.array(preds)
    ensemble_preds = w @ preds_stack
    objective = cp.sum(cp.abs(ensemble_preds - targets))
    
    # Apply constraints
    if constraints == 'simplex':
        constraints = [w >= 0, cp.sum(w) == 1]
    elif constraints == 'nonnegative':
        constraints = [w >= 0]
    elif constraints == 'none':
        constraints = []
    
    # Solve problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver='ECOS', verbose=verbose)
    return w.value


def solve_binary_logloss_probs(preds,
                               targets,
                               constraints='simplex',
                               verbose=False):
    '''
    Solve for optimal classifier ensemble using log loss (cross entropy)
    objective function. Ensembling is performed in the probability space.
    
    Args:
      preds: list or tuple of each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights (only 'simplex'
        is supported for this problem).
      verbose: whether CVXPY solver should generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    # Setup
    assert constraints == 'simplex'
    m = len(preds)
    
    # Get target probs
    preds_stack = np.array(preds)
    target_probs = preds_stack * targets + (1 - preds_stack) * (1 - targets)
    
    # Objectivew = cp.Variable(m)
    w = cp.Variable(m)
    ensemble_probs = w @ target_probs
    objective = - cp.sum(cp.log(ensemble_probs))
    
    # Simplex constraints
    constraints = [w >= 0, cp.sum(w) == 1]
    
    # Solve problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver='ECOS', verbose=verbose)
    return w.value


def solve_binary_logloss_logits(preds,
                                targets,
                                constraints='simplex',
                                verbose=False):
    '''
    Solve for optimal classifier ensemble using log loss (cross entropy)
    objective function. Ensembling is performed in the logit (log probability)
    space.

    Args:
      preds: list or tuple of each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights ('simplex',
        'nonnegative' or 'none').
      verbose: whether CVXPY solver should generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    # Setup
    assert constraints in ['simplex', 'nonnegative', 'none']
    m = len(preds)
    
    # Get target preds
    preds_stack = np.array(preds)
    target_logits = preds_stack * targets - preds_stack * (1 - targets)

    # Objective
    w = cp.Variable(m)
    ensemble_logits = w @ target_logits
    ensemble_logprobs = - cp.logistic(- ensemble_logits)
    objective = - cp.sum(ensemble_logprobs)
    
    # Apply constraints
    if constraints == 'simplex':
        constraints = [w >= 0, cp.sum(w) == 1]
    elif constraints == 'nonnegative':
        constraints = [w >= 0]
    elif constraints == 'none':
        constraints = []
    
    # Solve problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver='ECOS', verbose=verbose)
    return w.value


def solve_multiclass_logloss_probs(preds,
                                   targets,
                                   constraints='simplex',
                                   verbose=False):
    '''
    Solve for optimal classifier ensemble using log loss (cross entropy)
    objective function. Ensembling is performed in the probability space.
    
    Args:
      preds: list or tuple of each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights (only 'simplex'
        is supported for this problem).
      verbose: whether CVXPY solver should generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    # Setup
    assert constraints == 'simplex'
    n, k = preds[0].shape
    m = len(preds)

    # Get target probs
    preds_stack = np.array(preds)
    target_onehot = (np.arange(k)[np.newaxis].repeat(n, 0) == targets[:, np.newaxis]).astype(float)
    target_probs = (preds_stack * target_onehot).sum(axis=2)
    
    # Objective
    w = cp.Variable(m)
    ensemble_probs = w @ target_probs
    objective = - cp.sum(cp.log(ensemble_probs))
    
    # Simplex constraints
    constraints = [w >= 0, cp.sum(w) == 1]
    
    # Solve problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver='ECOS', verbose=verbose)
    return w.value


def solve_multiclass_logloss_logits(preds,
                                    targets,
                                    constraints='simplex',
                                    verbose=False):
    '''
    Solve for optimal classifier ensemble using log loss (cross entropy)
    objective function. Ensembling is performed in the logit (log probability)
    space.

    Args:
      preds: list or tuple of each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights ('simplex',
        'nonnegative' or 'none')
      verbose: whether CVXPY solver should generate verbose output.

    Returns: weights for optimal ensemble.
    '''
    # Setup
    assert constraints in ['simplex', 'nonnegative', 'none']
    n, k = preds[0].shape
    m = len(preds)
    
    # Get target, non-target preds
    preds_stack = np.array(preds)
    target_onehot = (np.arange(k)[np.newaxis].repeat(n, 0) == targets[:, np.newaxis]).astype(float)
    
    # Objective
    w = cp.Variable(m)
    ensemble_logits = cp.vstack([w @ preds_stack[:, :, i] for i in range(k)]).T
    target_logits = cp.sum(cp.multiply(ensemble_logits, target_onehot), axis=1)
    ensemble_logprobs = target_logits - cp.log_sum_exp(ensemble_logits, axis=1)
    objective = - cp.sum(ensemble_logprobs)

    # Apply constraints
    if constraints == 'simplex':
        constraints = [w >= 0, cp.sum(w) == 1]
    elif constraints == 'nonnegative':
        constraints = [w >= 0]
    elif constraints == 'none':
        constraints = []
    
    # Solve problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver='ECOS', verbose=verbose)
    return w.value
