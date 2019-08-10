from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
import sys


space = [
    hp.uniform('learningRate', 1e-5, 1e-3),
    # hp.uniform('priorityExponent', 0, 1),
    # hp.quniform('numIntermediateLayers', 1, 5, 1),
    # hp.quniform('intermediateLayerSize', 16, 128, 16),
    # hp.quniform('finalLayerSize', 64, 512, 64)
]

space = hp.uniform('learningRate', 2e-4, 2.5e-4)

def execTest(args):
    sys.path.append('/Users/dsmith11/repos/dqn/')
    from worker import objective
    return objective(args)

# space = hp.uniform('priorityExponent', 0, 1)
# trials = MongoTrials('mongo://localhost:1234/dqn-optimization/jobs', exp_key='lunar-lander-prioritization')
best = fmin(fn=execTest, space=space, algo=tpe.suggest, max_evals=100, verbose=1)
print(best)