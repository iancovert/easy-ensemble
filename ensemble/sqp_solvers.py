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
      preds: an iterable (e.g., list, tuple) over each model's predictions.
      targets: prediction targets.
      helper_fn: helper function to calculate objective, grads, hessian.
      max_iters: max number of iterations (Newton/SQP steps).
      tolerance: for detecting convergence.
      verbose: whether to generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    # Initialization
    m = len(preds)
    w = np.ones(m) / m
    prev_objective, grad, hess = helper_fn(preds, targets, w)
    
    # For tracking best solution
    best_solution = w
    best_objective = prev_objective
    
    # Begin optimizing
    converged = False
    for it in range(max_iters):
        # Take Newton step
        w = w - np.linalg.solve(hess, grad)
        
        # Calculate objective, check if converged
        objective, grad, hess = helper_fn(preds, targets, w)
        if verbose:
            print(f'Objective after step {it + 1}: {objective}')
        if objective < best_objective:
            best_objective = objective
            best_solution = w
        else:
            print('Solution got worse!')
        if (prev_objective - objective) / prev_objective < tolerance:
            converged = True
            if verbose:
                print(f'Stopping after {it + 1} steps')
            break
        else:
            prev_objective = objective

    # Return result
    if not converged:
        print(f'Did not converge within {max_iters} steps, solution may be inexact')

    return best_solution


def sqp_solver(preds,
               targets,
               constraints,
               helper_fn,
               max_iters=100,
               tolerance=1e-5,
               verbose=False):
    '''
    Solve for optimal ensemble using SQP.
    
    Args:
      preds: an iterable (e.g., list, tuple) over each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights ('simplex' or
        'nonnegative').
      helper_fn: helper function to calculate objective, grads, hessian.
      max_iters: max number of iterations (Newton/SQP steps).
      tolerance: for detecting convergence.
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
    
    # For tracking best solution
    best_solution = w
    best_objective = prev_objective
    
    # Begin optimizing
    converged = False
    for it in range(max_iters):
        # QP setup
        P = sparse.csc_matrix(hess)
        q = grad - P @ w
        
        # Solve problem
        problem = osqp.OSQP()
        problem.setup(P, q, A, l, u, verbose=verbose)
        solution = problem.solve()
        w = solution.x
        
        # Calculate objective, check if converged
        objective, grad, hess = helper_fn(preds, targets, w)
        if verbose:
            print(f'Objective after step {it + 1}: {objective}')
        if objective < best_objective:
            best_objective = objective
            best_solution = w
        else:
            print('SQP solution got worse')
        if (prev_objective - objective) / prev_objective < tolerance:
            converged = True
            if verbose:
                print(f'Stopping after {it + 1} steps')
            break
        else:
            prev_objective = objective

    # Return result
    if not converged:
        print(f'Did not converge within {max_iters} steps, solution may be inexact')

    return best_solution


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
                               verbose=False):
    '''
    Solve for optimal classifier ensemble using log loss (cross entropy)
    objective function. Ensembling is performed in the probability space.
    
    Args:
      preds: an iterable (e.g., list, tuple) over each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights (only 'simplex'
        is supported for this problem).
      max_iters: max number of iterations (Newton/SQP steps).
      tolerance: for detecting convergence.
      verbose: whether to generate verbose output.
      
    Returns: weights for optimal ensemble.
    '''
    assert constraints == 'simplex'
    return sqp_solver(
        preds, targets, constraints, binary_logloss_probs_helper, max_iters,
        tolerance, verbose)


def binary_logloss_logits_helper(preds, targets, w):
    '''Helper function to calculate objective, grads and hessian.'''
    preds_stack = np.array(preds)
    target_logits = preds_stack * targets - preds_stack * (1 - targets)
    ensemble_logits = w @ target_logits
    ensemble_probs = 1 / (1 + np.exp(- ensemble_logits))
    objective = - np.sum(np.log(ensemble_probs))
    grad = - np.sum(target_logits * (1 - ensemble_probs), axis=1)
    hess = np.sum(target_logits * target_logits[:, np.newaxis] * ensemble_probs * (1 - ensemble_probs), axis=2)
    return objective, grad, hess


def solve_binary_logloss_logits(preds,
                                targets,
                                constraints='simplex',
                                max_iters=100,
                                tolerance=1e-6,
                                verbose=False):
    '''
    Solve for optimal classifier ensemble using log loss (cross entropy)
    objective function. Ensembling is performed in the logit (log probability)
    space.

    Args:
      preds: an iterable (e.g., list, tuple) over each model's predictions.
      targets: prediction targets.
      constraints: constraints for learned ensemble weights ('simplex',
        'nonnegative' or 'none').
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
            max_iters, tolerance, verbose)
