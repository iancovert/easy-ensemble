import osqp
import numpy as np
from scipy import sparse


def newton_solver(preds,
                  targets,
                  helper_fn,
                  max_iters=100,
                  tolerance=1e-5,
                  verbose=False):
    '''
    Solve for optimal ensemble using Newton's algorithm.
    
    Args:
      preds: list or tuple of each model's predictions.
      targets: prediction targets.
      helper_fn: helper function to calculate objective, grads, hessian.
      max_iters: max number of Newton iterations.
      tolerance: threshold for terminating SQP/Newton iterations.
      verbose: whether to generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    # Initialization
    m = len(preds)
    w = np.ones(m) / m
    prev_objective, grad, hess = helper_fn(preds, targets, w)
    
    # Begin optimizing
    converged = False
    for it in range(max_iters):
        # Take Newton step
        w = w - np.linalg.solve(hess, grad)
        
        # Calculate objective, check if converged
        objective, grad, hess = helper_fn(preds, targets, w)
        if verbose:
            print(f'Objective after step {it + 1}: {objective}')
        if (prev_objective - objective) / prev_objective < tolerance:
            converged = True
            if verbose:
                print(f'Stopping after {it + 1} steps')
            break
        prev_objective = objective

    # Return result
    if not converged and max_iters > 1:
        print(f'Did not converge within {max_iters} steps, solution may be inexact')

    return w


def sqp_solver(preds,
               targets,
               constraints,
               helper_fn,
               max_iters=100,
               tolerance=1e-5,
               eps_rel=1e-8,
               verbose=False):
    '''
    Solve for optimal ensemble using SQP.
    
    Args:
      preds: list or tuple of each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights ('simplex' or
        'nonnegative').
      helper_fn: helper function to calculate objective, grads, hessian.
      max_iters: max number of SQP iterations.
      tolerance: threshold for terminating SQP/Newton iterations.
      eps_rel: relative tolerance for SQP solution.
      verbose: whether to generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    # Initialization
    m = len(preds)
    w = np.ones(m) / m
    prev_objective, grad, hess = helper_fn(preds, targets, w)
    
    # Constraints
    if constraints == 'simplex':
        A = sparse.csc_matrix(np.concatenate([np.ones(m)[np.newaxis], np.eye(m)], axis=0))
        l = np.zeros(m + 1)
        l[0] = 1
        u = np.ones(m + 1)
    elif constraints == 'nonnegative':
        A = sparse.csc_matrix(np.eye(m))
        l = np.zeros(m)
        u = np.inf * np.ones(m)
    
    # Begin optimizing
    converged = False
    for it in range(max_iters):
        # QP setup
        P = sparse.csc_matrix(hess)
        q = grad - P @ w
        
        # Solve problem
        problem = osqp.OSQP()
        problem.setup(P, q, A, l, u, verbose=verbose, eps_rel=eps_rel)
        solution = problem.solve()
        w = solution.x
        
        # Calculate objective, check if converged
        objective, grad, hess = helper_fn(preds, targets, w)
        if verbose:
            print(f'Objective after step {it + 1}: {objective}')
        if (prev_objective - objective) / prev_objective < tolerance:
            converged = True
            if verbose:
                print(f'Stopping after {it + 1} steps')
            break
        prev_objective = objective

    # Return result
    if not converged and max_iters > 1:
        print(f'Did not converge within {max_iters} steps, solution may be inexact')

    return w


def regressor_mse_helper(preds, targets, w):
    '''Helper function to calculate objective, grads and hessian.'''
    preds_stack = np.array(preds)
    ensemble_preds = w @ preds_stack
    residuals = ensemble_preds - targets
    objective = residuals @ residuals
    grad = 2 * preds_stack @ residuals
    hess = 2 * preds_stack @ preds_stack.T
    return objective, grad, hess


def solve_regressor_mse(preds,
                        targets,
                        constraints='simplex',
                        max_iters=1,
                        tolerance=1e-6,
                        eps_rel=1e-8,
                        verbose=False):
    '''
    Solve for optimal regressor ensemble using MSE objective function.

    Args:
      preds: list or tuple of each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights ('simplex',
        'nonnegative' or 'none').
      max_iters: max number of SQP/Newton iterations. Only a single iteration
        will be used regardless of the value.
      tolerance: threshold for terminating SQP/Newton iterations.
      eps_rel: relative tolerance for SQP solution.
      verbose: whether to generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    assert constraints in ['simplex', 'nonnegative', 'none']
    if constraints == 'none':
        return newton_solver(
            preds, targets, regressor_mse_helper, 1, tolerance,
            verbose)
    else:
        return sqp_solver(
            preds, targets, constraints, regressor_mse_helper, 1,
            tolerance, eps_rel, verbose)


def binary_logloss_probs_helper(preds, targets, w):
    '''Helper function to calculate objective, grads and hessian.'''
    preds_stack = np.array(preds)
    target_probs = preds_stack * targets + (1 - preds_stack) * (1 - targets)
    ensemble_probs = w @ target_probs
    objective = - np.sum(np.log(ensemble_probs))
    grad = - np.sum(target_probs / ensemble_probs, axis=1)
    hess = np.sum(target_probs * target_probs[:, np.newaxis] / ensemble_probs ** 2, axis=2)
    return objective, grad, hess


def solve_binary_logloss_probs(preds,
                               targets,
                               constraints='simplex',
                               max_iters=100,
                               tolerance=1e-6,
                               eps_rel=1e-8,
                               verbose=False):
    '''
    Solve for optimal classifier ensemble using log loss (cross entropy)
    objective function. Ensembling is performed in the probability space.
    
    Args:
      preds: list or tuple of each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights (only 'simplex'
        is supported for this problem).
      max_iters: max number of SQP/Newton iterations.
      tolerance: threshold for terminating SQP/Newton iterations.
      eps_rel: relative tolerance for SQP solution.
      verbose: whether to generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    assert constraints == 'simplex'
    return sqp_solver(
        preds, targets, constraints, binary_logloss_probs_helper, max_iters,
        tolerance, eps_rel, verbose)


def binary_logloss_logits_helper(preds, targets, w):
    '''Helper function to calculate objective, grads and hessian.'''
    preds_stack = np.array(preds)
    target_logits = preds_stack * targets - preds_stack * (1 - targets)
    ensemble_logits = w @ target_logits
    ensemble_probs = 1 / (1 + np.exp(- ensemble_logits))
    objective = - np.sum(np.log(ensemble_probs))
    grad = - target_logits @ (1 - ensemble_probs)
    hess = np.sum(target_logits * target_logits[:, np.newaxis] * ensemble_probs * (1 - ensemble_probs), axis=2)
    return objective, grad, hess


def solve_binary_logloss_logits(preds,
                                targets,
                                constraints='simplex',
                                max_iters=100,
                                tolerance=1e-6,
                                eps_rel=1e-8,
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
      max_iters: max number of SQP/Newton iterations.
      tolerance: threshold for terminating SQP/Newton iterations.
      eps_rel: relative tolerance for SQP solution.
      verbose: whether to generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    assert constraints in ['simplex', 'nonnegative', 'none']
    if constraints == 'none':
        return newton_solver(
            preds, targets, binary_logloss_logits_helper, max_iters, tolerance,
            verbose)
    else:
        return sqp_solver(
            preds, targets, constraints, binary_logloss_logits_helper,
            max_iters, tolerance, eps_rel, verbose)


def multiclass_logloss_probs_helper(preds, targets, w):
    '''Helper function to calculate objective, grads and hessian.'''
    n, k = preds[0].shape
    preds_stack = np.array(preds)
    target_onehot = (np.arange(k)[np.newaxis].repeat(n, 0) == targets[:, np.newaxis]).astype(float)
    target_probs = (preds_stack * target_onehot).sum(axis=2)
    ensemble_probs = w @ target_probs
    objective = - np.sum(np.log(ensemble_probs))
    grad = - np.sum(target_probs / ensemble_probs, axis=1)
    hess = np.sum(target_probs * target_probs[:, np.newaxis] / ensemble_probs ** 2, axis=2)
    return objective, grad, hess


def solve_multiclass_logloss_probs(preds,
                                   targets,
                                   constraints='simplex',
                                   max_iters=100,
                                   tolerance=1e-6,
                                   eps_rel=1e-8,
                                   verbose=False):
    '''
    Solve for optimal classifier ensemble using log loss (cross entropy)
    objective function. Ensembling is performed in the probability space.
    
    Args:
      preds: list or tuple of each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights (only 'simplex'
        is supported for this problem).
      max_iters: max number of SQP/Newton iterations.
      tolerance: threshold for terminating SQP/Newton iterations.
      eps_rel: relative tolerance for SQP solution.
      verbose: whether to generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    assert constraints == 'simplex'
    return sqp_solver(
        preds, targets, constraints, multiclass_logloss_probs_helper, max_iters,
        tolerance, eps_rel, verbose)



def multiclass_logloss_logits_helper(preds, targets, w):
    '''Helper function to calculate objective, grads and hessian.'''
    n, k = preds[0].shape
    preds_stack = np.array(preds)
    target_onehot = (np.arange(k)[np.newaxis].repeat(n, 0) == targets[:, np.newaxis]).astype(float)
    ensemble_logits = (preds_stack.T @ w).T
    ensemble_probs = np.exp(ensemble_logits) / np.sum(np.exp(ensemble_logits), axis=1, keepdims=True)
    target_logprobs = np.log(np.sum(ensemble_probs * target_onehot, axis=1))
    objective = - np.sum(target_logprobs)
    grad = - np.sum(preds_stack * (target_onehot - ensemble_probs), axis=(1, 2))
    temp = preds_stack.swapaxes(0, 1) @ ensemble_probs[:, :, np.newaxis]
    hess = np.sum(
        preds_stack.swapaxes(0, 1) @ (preds_stack.swapaxes(1, 2) * ensemble_probs.T).T
        - temp @ temp.swapaxes(1, 2), axis=0)
    # Another way of writing hessian
    # hess = np.sum(
    #     preds_stack.swapaxes(0, 1)
    #     @ ((np.eye(k) * ensemble_probs[:, np.newaxis])
    #         - (ensemble_probs.T * ensemble_probs.T[:, np.newaxis]).T)
    #     @ preds_stack.swapaxes(1, 2).T, axis=0)
    return objective, grad, hess


def solve_multiclass_logloss_logits(preds,
                                    targets,
                                    constraints='simplex',
                                    max_iters=100,
                                    tolerance=1e-6,
                                    eps_rel=1e-8,
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
      max_iters: max number of SQP/Newton iterations.
      tolerance: threshold for terminating SQP/Newton iterations.
      eps_rel: relative tolerance for SQP solution.
      verbose: whether to generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    assert constraints in ['simplex', 'nonnegative', 'none']
    if constraints == 'none':
        return newton_solver(
            preds, targets, multiclass_logloss_logits_helper, max_iters, tolerance,
            verbose)
    else:
        return sqp_solver(
            preds, targets, constraints, multiclass_logloss_logits_helper,
            max_iters, tolerance, eps_rel, verbose)
