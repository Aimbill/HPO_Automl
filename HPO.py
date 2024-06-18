import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
from .utils import get_logger

logger = get_logger(__name__)

class HyperoptOptimizer:
    def __init__(self, models, aim_grids, max_evals=50):
        self.models = models
        self.aim_grids = aim_grids
        self.max_evals = max_evals
        self.best_scores = {model_name: -np.inf for model_name, _ in models}
        self.best_aim = {model_name: None for model_name, _ in models}
        self.best_models = {model_name: None for model_name, _ in models}
        self.best_model_name = None

    def evaluate(self, model, X, y, aim, problem_type):
        model.set_aim(**aim)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy' if problem_type == 'classification' else 'neg_mean_squared_error')
        return np.mean(scores)

    def objective(self, aim):
        model_idx = aim.pop('model')
        model_name, model = self.models[model_idx]
        model_aim = {k.split('__', 1)[-1]: v for k, v in aim.items()}

        valid_aim = model.get_aim().keys()
        model_aim = {k: v for k, v in model_aim.items() if k in valid_aim}

        try:
            model.set_aim(**model_aim)
            scores = cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error')
            score = np.mean(scores)
        except Exception as e:
            logger.error(f"Error with model {model_name} and aim {model_aim}: {e}")
            return {'loss': np.inf, 'status': STATUS_OK}

        if score > self.best_scores[model_name]:
            self.best_scores[model_name] = score
            self.best_models[model_name] = model
            self.best_aim[model_name] = model_aim
            # Track the overall best model across all evaluations
            if self.best_model_name is None or score > self.best_scores[self.best_model_name]:
                self.best_model_name = model_name

        return {'loss': -score, 'status': STATUS_OK}
    def suggest(self, l, g):
        gamma = 0.25
        n = len(l) + len(g)
        top_n = max(1, int(np.floor(gamma * n)))
        l_sorted = sorted(l, key=lambda x: x[1], reverse=True)
        g_sorted = sorted(g, key=lambda x: x[1], reverse=True)
        selected = np.random.choice(l_sorted[:top_n] + g_sorted[:top_n])
        return selected[0]

    def optimize(self, X, y, problem_type):
        self.X = X
        self.y = y
        self.problem_type = problem_type

        space = {
            'model': hp.choice('model', range(len(self.models)))
        }

        for model_name, aim_grid in self.aim_grids.items():
            for aim, values in aim_grid.items():
                aim_name = f'{model_name}__{aim}'
                if isinstance(values, list):
                    space[aim_name] = hp.choice(aim_name, values)
                elif isinstance(values, dict):
                    if 'dist' in values and values['dist'] == 'uniform':
                        space[aim_name] = hp.uniform(aim_name, values['low'], values['high'])
                    elif 'dist' in values and values['dist'] == 'normal':
                        space[aim_name] = hp.normal(aim_name, values['loc'], values['scale'])
        
        trials = Trials()
        fmin(self.objective, space=space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)

        best_model_name = self.best_model_name
        return self.best_models[best_model_name], self.best_aim[best_model_name], self.best_scores[best_model_name]
