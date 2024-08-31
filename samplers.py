from typing import Any, Iterable
import numpy as np
from scipy import stats


class EmpiricalSampler:
    def __init__(self, variable: int, premium: bool):
        if variable == 0: #price
            self.dists = [
                stats.cauchy(loc=0.6360821940299033, scale=0.04049838857838751),
                stats.cauchy(loc=0.6961574067798582, scale=0.0706278515565138),
                stats.exponpow(loc=-0.8437255275578176, scale=1.6776490426869586, b=8.606679476346438),
                stats.cauchy(loc=0.6981269242106626, scale=0.035425072079324625),
                stats.cauchy(loc=0.6695189081656098, scale=0.05518520525307997),
                stats.uniform(0, 1),
            ]
        elif variable == 1: #delivery
            self.dists = [
                stats.exponpow(loc=-6689473.777250121, scale=6689474.709018325, b=66810461.11149461),
                stats.cauchy(loc=0.7409407481917889, scale=0.07924884967278138),
                stats.cauchy(loc=0.6720358608469594, scale=0.09244661138766927),
                stats.cauchy(loc=0.8266660564860485, scale=0.05139501487147483),
                stats.exponpow(loc=-3.790698954715651, scale=4.672268681787388, b=27.09683074665709),
                stats.uniform(0, 1),
            ]
        elif variable == 2: #installment
            self.dists = [
                stats.cauchy(loc=0.47163313839170623, scale=0.06798057256484127),
                stats.exponpow(loc=-1.3600063616426152, scale=2.1855178028216726, b=11.879739050405925),
                stats.exponpow(loc=-1.6878879075118047, scale=2.506123826150217, b=11.323994432249123),
                stats.uniform(scale=1, loc=0),
                stats.cauchy(loc=0.37803494711915464, scale=0.10791293487722513),
                stats.uniform(0, 1),
            ]
        else: #seller score
            self.dists = [
                stats.exponpow(loc=-0.0175683237234851, scale=0.8739227106896492, b=0.7130753129518563),
                stats.exponpow(loc=-0.451979405391409, scale=1.319905962869526, b=3.119621364298831),
                stats.exponpow(loc=-0.12061719066843017, scale=0.9660544040078557, b=1.4935101073026793),
                stats.exponpow(loc=-0.12061719066843017, scale=0.9660544040078557, b=1.4935101073026793),
                stats.exponpow(loc=-0.0543237526278426, scale=0.7764842451896865, b=0.8537792031634603),
                stats.uniform(0, 1),
            ]

    def sample(self, types: Iterable):
        sample_list = []
        for tp in types:
            generated_value = self.dists[tp].rvs(size=1)
            while generated_value > 1 or generated_value <= 0:
                generated_value = self.dists[tp].rvs(size=1)
            sample_list.append(generated_value[0])
        return sample_list

class NormSampler:
    def __init__(self, low: Any, high: Any, dtype: type, loc: Any, scale: Any):
        self.loc = loc
        self.scale = scale
        self.dist = stats.truncnorm(a=low, b=high, loc=loc, scale=scale)
        self.dtype = dtype

    def sample(self, size: int) -> np.array:
        return self.dist.rvs(size=size).astype(self.dtype)

class ChoiceSampler:
    def __init__(self, options: tuple, p: tuple):
        self.options = np.array(options)
        self.p = p

    def sample(self, size: int) -> np.array:
        return np.random.choice(self.options, size, p=self.p)
