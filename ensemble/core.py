import numpy as np
from ensemble import sqp_solvers, utils


SOLVER_DICT = {
    'reg:mse': sqp_solvers.solve_regressor_mse,
    'binary:logloss_probs': sqp_solvers.solve_binary_logloss_probs,
    'binary:logloss_logits': sqp_solvers.solve_binary_logloss_logits,
    'multi:logloss_probs': sqp_solvers.solve_multiclass_logloss_probs,
    'multi:logloss_logits': sqp_solvers.solve_multiclass_logloss_logits
}


def verify_arguments(preds, targets, objective):
    # Verify shapes
    shape = preds[0].shape
    for pred in preds[1:]:
        assert pred.shape == shape, '[Easy-Ensemble]: Predictions must all have same shape'
    assert targets.shape == (shape[0],), '[Easy-Ensemble]: Target shape is incorrect'
    
    # Verify shapes for specific objectives
    if objective in ['multi:logloss_probs', 'multi:logloss_logits']:
        assert len(preds[0].shape) == 2, f'[Easy-Ensemble]: Predictions should have two dimensions for objective `{objective}`'
    else:
        assert len(preds[0].shape) == 1, f'[Easy-Ensemble]: Predictions should have one dimension for objective `{objective}`'
    
    # Verify values for specific objective
    if objective in ['binary:logloss_probs', 'binary:logloss_logits',
                     'multi:logloss_probs', 'multi:logloss_logits']:
        for pred in preds:
            assert pred.min() >= 0 and pred.max() <= 1, f'[Easy-Ensemble]: Predictions must be probabilities for objective `{objective}`'
            


class Ensemble:
    '''
    Ensemble with learned weights.
    
    Args:
      objective: optimization objective ('reg:mse', 'binary:logloss_probs',
        'binary:logloss_logits', 'multi:logloss_probs', 'multi:logloss_logits')
      constraints: constraints for learned ensemble weights ('simplex',
        'nonnegative' or 'none')
      max_iters: max number of SQP/Newton iterations.
      tolerance: threshold for terminating SQP/Newton iterations.
      eps_rel: relative tolerance for SQP solution.
      verbose: whether solver should generate verbose output.
    '''

    def __init__(self,
                 objective,
                 constraints='simplex',
                 max_iters=100,
                 tolerance=1e-6,
                 eps_rel=1e-8,
                 verbose=False):
        # Verify objective and set solver
        assert objective in SOLVER_DICT.keys(), f'[Easy-Ensemble]: Unrecognized objective `{objective}`'
        self.solver = SOLVER_DICT[objective]
        
        # Verify constraints
        if objective in [
            'binary:logloss_probs',
            'multi:logloss_probs'
        ]:
            assert constraints in ['simplex'], f'[Easy-Ensemble]: Constraints compatible with objective `{objective}` are `simplex`'
        else:
            assert constraints in ['simplex', 'nonnegative', 'none'], f'[Easy-Ensemble]: Constraints compatible with objective `{objective}` are `simplex`, `nonnegative`, `none`'
        self.constraints = constraints
        
        # Set input/output transforms
        if objective == 'multi:logloss_logits':
            self.input_transform = utils.stable_log
            self.output_transform = utils.softmax
        elif objective == 'binary:logloss_logits':
            self.input_transform = utils.stable_logit
            self.output_transform = utils.sigmoid
        else:
            self.input_transform = None
            self.output_transform = None
            
        # For solver
        self.objective = objective
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.eps_rel = eps_rel
        self.verbose = verbose
    
    def fit(self, preds, targets):
        '''
        Fit the learned ensemble.
        
        Args:
          preds: iterable (list, tuple) over each model's predictions.
          targets: prediction targets.
        '''
        # Verify arguments
        verify_arguments(list(preds), targets, self.objective)

        # Apply input transform
        if self.input_transform:
            preds = [self.input_transform(preds) for preds in preds]

        # Find optimal weights
        self.weights = self.solver(
            list(preds), targets, self.constraints, self.max_iters,
            self.tolerance, self.eps_rel, self.verbose)
        return self

    def predict(self, preds):
        '''
        Apply the learned ensemble.
        
        Args:
          preds: iterable (list, tuple) over each model's predictions.
        '''
        # Apply input transform
        if self.input_transform:
            preds = [self.input_transform(preds) for preds in preds]
            
        # Default to evenly weighted ensemble
        if not hasattr(self, 'weights'):
            print('[Easy-Ensemble]: Defaulting to evenly weighted ensemble. '
                  'Use the `fit` function to optimize the ensemble')
            self.weights = np.ones(len(preds)) / len(preds)
        
        # Apply learned ensemble
        return utils.apply_ensemble(
            list(preds), self.weights, self.output_transform)
        
    def get_weights(self):
        return self.weights
