import time
from abc import ABC

import numpy
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import optimization
import scipy.optimize as optim
from model import OptimizationModel

KL_DIVERGENCE = "rKL"  # represent kl-divergence group fairness measure
ND_DIFFERENCE = "rND"  # represent normalized difference group fairness measure
RD_DIFFERENCE = "rRD"  # represent ratio difference group fairness measure


class Agent(ABC):

    def __init__(self):
        pass

    def rank(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class LinearAgent(Agent):
    def __init__(self, weigths: list):
        self.weights = weigths

    def rank(self, data: pd.DataFrame) -> pd.DataFrame:
        scores = []
        scores_baseline = []
        for i in range(len(data.index)):
            row = data.loc[i, :].values.flatten().tolist()
            row = row[:-1]
            assert len(self.weights) == len(row)
            score = 0
            score_baseline = 0
            for idx in range(len(row)):
                score += self.weights[idx] * float(row[idx])
                if idx < len(row) - 1:
                    score_baseline += self.weights[idx] * float(row[idx])
            scores.append(score)
            scores_baseline.append(score_baseline)
        ranking_reference = list(reversed(range(1, len(scores) + 1)))
        rank_list = rankdata(scores, method='ordinal')
        rank_baseline_list = rankdata(scores_baseline, method='ordinal')
        reversed_rank_list = []
        reversed_rank_baseline_list = []
        for i in range(len(scores)):
            reversed_rank_list.append(ranking_reference[rank_list[i] - 1])
            reversed_rank_baseline_list.append(ranking_reference[rank_baseline_list[i] - 1])
        data['scores'] = scores
        data['rank'] = reversed_rank_list
        data['scores_baseline'] = scores_baseline
        data['rank_baseline'] = reversed_rank_baseline_list
        return data


class FairAgent(Agent):
    def __init__(self, model: OptimizationModel):
        self.model = model

    def rank(self, data: pd.DataFrame):
        prep_data = data.drop(['premium', 'rank', 'scores'], axis=1)
        prep_data = prep_data.to_numpy()
        predictions = self.model.predict(prep_data)
        ranking_reference = list(reversed(range(1, len(predictions) + 1)))
        rank_list = rankdata(predictions, method='ordinal')
        reversed_rank_list = []
        for i in range(len(predictions)):
            reversed_rank_list.append(ranking_reference[rank_list[i] - 1])
        data['fair_scores'] = predictions
        data['fair_rank'] = reversed_rank_list
        return data


class FairAgentOptimizer(Agent):
    def __init__(self):
        pass

    def rank(self, data: pd.DataFrame):
        sensi_att = data['premium'].to_numpy()
        input_scores = data['scores'].to_numpy()
        prep_data = data.drop(['premium', 'rank', 'scores'], axis=1)
        prep_data = prep_data.to_numpy()
        pro_index = np.array(np.where(sensi_att == False))[0].flatten()
        unpro_index = np.array(np.where(sensi_att != False))[0].flatten()
        pro_data = prep_data[pro_index, :]
        unpro_data = prep_data[unpro_index, :]
        _accmeasure = "scoreDiff"
        _k = 4

        start_time = time.time()
        print("Starting optimization @ ", _k, "ACCM ", _accmeasure, " time: ", start_time)
        # initialize the optimization
        rez, bnd = optimization.initOptimization(prep_data, _k)
        rez = optim.fmin_l_bfgs_b(optimization.lbfgsOptimize, x0=rez, disp=1, epsilon=1e-5,
                                  args=(prep_data, pro_data, unpro_data, input_scores, _accmeasure, _k, 0.01,
                                             1, 100, 0), bounds=bnd, approx_grad=True, factr=1e12, pgtol=1e-04, maxfun=15,
                                       maxiter=50)
        end_time = time.time()
        print("Ending optimization ", "@ ", " warnflag ", rez[2]['warnflag'], _k, "ACCM ", _accmeasure, " time: ", end_time)
        model = OptimizationModel(rez, _k)
        predictions = model.predict(prep_data)
        data['fair_scores'] = predictions
        ranking_reference = list(reversed(range(1, len(predictions) + 1)))
        rank_list = rankdata(predictions, method='ordinal')
        reversed_rank_list = []
        for i in range(len(predictions)):
            reversed_rank_list.append(ranking_reference[rank_list[i] - 1])
        data['fair_rank'] = reversed_rank_list
        return data
