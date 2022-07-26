import osqp
import numpy as np
from scipy import sparse


def binary_logloss_probs_objective(target_probs, w):
    ensemble_probs = w @ target_probs
    return - np.sum(np.log(ensemble_probs))


def binary_logloss_probs_helper(target_probs, w):
    ensemble_probs = w @ target_probs
    grad = - np.sum(target_probs / ensemble_probs, axis=1)
    hess = np.sum(target_probs * target_probs[:, np.newaxis] / ensemble_probs ** 2, axis=2)
    return grad, hess


def solve_binary_logloss_probs(preds,
                               targets,
                               constraints='simplex',
                               max_iters=100,
                               tolerance=1e-5,
                               verbose=False):
    # Setup
    assert constraints == 'simplex'
    m = len(preds)
    preds_stack = np.array(preds)
    target_probs = preds_stack * targets + (1 - preds_stack) * (1 - targets)
    
    # Initialization
    w = np.ones(m) / m
    prev_objective = binary_logloss_probs_objective(target_probs, w)
    
    # For tracking best solution
    best_solution = w
    best_objective = prev_objective
    
    # Constraints
    A = sparse.csc_matrix(np.concatenate([np.ones(m)[np.newaxis], np.eye(m)], axis=0))
    l = np.zeros(m + 1)
    l[0] = 1
    u = np.ones(m + 1)
    
    # Solve SQP
    converged = False
    for it in range(max_iters):
        # QP setup
        grad, hess = binary_logloss_probs_helper(target_probs, w)
        P = sparse.csc_matrix(hess)
        q = grad - P @ w
        
        # Solve problem
        problem = osqp.OSQP()
        problem.setup(P, q, A, l, u, verbose=verbose)
        solution = problem.solve()
        w = solution.x
        
        # Calculate objective, check if converged
        objective = binary_logloss_probs_objective(target_probs, w)
        if verbose:
            print(f'Objective after step {it + 1}: {objective}')
        if objective < best_objective:
            best_objective = objective
            best_solution = w
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


# TODO for maximum consistency and reusability of code, should these helper
# functions always accept the same arguments as the solver?
# Recalculating same terms is bad, but perhaps negligible relative to QP solve
def binary_logloss_logits_objective(target_preds, w):
    ensemble_preds = w @ target_preds
    ensemble_probs = 1 / (1 + np.exp(- ensemble_preds))
    return - np.sum(np.log(ensemble_probs))


def binary_logloss_logits_helper(target_preds, w):
    ensemble_preds = w @ target_preds
    ensemble_probs = 1 / (1 + np.exp(- ensemble_preds))
    grad = - np.sum(target_preds * (1 - ensemble_probs), axis=1)
    hess = np.sum(target_preds * target_preds[:, np.newaxis] * ensemble_probs * (1 - ensemble_probs), axis=2)
    return grad, hess


def solve_binary_logloss_logits(preds,
                                targets,
                                constraints='simplex',
                                max_iters=100,
                                tolerance=1e-5,
                                verbose=False):
    # Setup
    assert constraints in ['simplex', 'nonnegative', 'none']
    m = len(preds)
    preds_stack = np.array(preds)
    target_preds = preds_stack * targets + (1 - preds_stack) * (1 - targets)
    
    # Initialization
    w = np.ones(m) / m
    prev_objective = binary_logloss_logits_objective(target_preds, w)
    
    # For tracking best solution
    best_solution = w
    best_objective = prev_objective
    
    # Constraints
    if constraints == 'simplex':
        A = sparse.csc_matrix(np.concatenate([np.ones(m)[np.newaxis], np.eye(m)], axis=0))
        l = np.zeros(m + 1)
        l[0] = 1
        u = np.ones(m + 1)
        constrained = True
    elif constraints == 'nonnegative':
        A = sparse.csc_matrix(np.eye(m))
        l = np.zeros(m)
        u = np.inf * np.ones(m)
        constrained = True
    else:
        constrained = False
    
    # Solve SQP
    converged = False
    for it in range(max_iters):
        # Problem setup
        grad, hess = binary_logloss_logits_helper(target_preds, w)
        if constrained:
            # Solve QP
            P = sparse.csc_matrix(hess)
            q = grad - P @ w
            
            # Solve problem
            problem = osqp.OSQP()
            problem.setup(P, q, A, l, u, verbose=verbose)
            solution = problem.solve()
            w = solution.x
            
        else:
            # Take Newton step
            w = w - np.linalg.solve(hess, grad)
        
        # Calculate objective, check if converged
        objective = binary_logloss_logits_objective(target_preds, w)
        if verbose:
            print(f'Objective after step {it + 1}: {objective}')
        if objective < best_objective:
            best_objective = objective
            best_solution = w
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
