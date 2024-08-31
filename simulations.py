from math import ceil, sqrt, pow
from typing import Any
import random
import pandas as pd
import numpy as np

from samplers import EmpiricalSampler, ChoiceSampler
from metrics import calculateNDFairness, getNormalizer, calculatePositionDifference, calculateScoreDifference, calculateKendallDistance
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.preprocessing import minmax_scale
from wonderwords import RandomWord
from agents import Agent
from log import LoggerData

COLUMNS = ['price', 'delivery_time', 'seller_score', 'installment_ratio', 'premium', 'cluster']


class Seller:
    def __init__(self,
                 name: str,
                 premium: bool,
                 price: float,
                 delivery_time: float,
                 seller_score: int,
                 cluster: int,
                 installment_ratio: float
                 ):
        # attributes
        self.participations = 0
        self.position_acc = 0
        self.premium = premium
        self.price = price
        self.name = name
        self.cluster = cluster
        self.delivery_time = delivery_time
        self.seller_score = seller_score
        self.installment_ratio = installment_ratio

    @property
    def ordered_list(self) -> list:
        return [self.price, self.delivery_time, self.seller_score,
                self.installment_ratio, self.premium, self.cluster]

    @property
    def mean_position(self) -> float:
        if float(self.participations) > 0:
            return float(self.position_acc) / float(self.participations)
        else:
            return 0.0


def name_generator() -> str:
    name = RandomWord().word(include_parts_of_speech=["nouns"])
    name += " "
    name += RandomWord().word(include_parts_of_speech=["adjective"])
    return name


class SellerPool:
    def __init__(self,
                 size: int,
                 premium_ratio: float,
                 tradeoff_ratio: int,
                 price_sampler: EmpiricalSampler,
                 delivery_time_sampler: EmpiricalSampler,
                 seller_score_sampler: EmpiricalSampler,
                 installment_sampler: EmpiricalSampler,
                 reset_each: int,
                 premium_seller_score_sampler: EmpiricalSampler,
                 ):
        # attributes
        self.size = size
        self.reset_each = reset_each
        self.premium_ratio = premium_ratio
        self.tradeoff_ratio = tradeoff_ratio
        self.price_sampler = price_sampler
        self.delivery_time_sampler = delivery_time_sampler
        self.seller_score_sampler = seller_score_sampler
        self.installment_sampler = installment_sampler
        self.premium_seller_score_sampler = premium_seller_score_sampler
        self.premium_size = int(ceil(self.size * self.premium_ratio))
        self.product_select = ChoiceSampler((0, 1, 2, 3, 4, 5), (.21, .16, .08, .20, .12, .23))

        self.price_acc = 0
        self.delivery_time_acc = 0
        self.price_fair_acc = 0
        self.delivery_time_fair_acc = 0
        self.price_baseline_acc = 0
        self.delivery_time_baseline_acc = 0

        self.position_dist_acc = 0
        self.kendall_dist_acc = 0
        self.spearman_dist_acc = 0
        self.score_diff_acc = 0
        self.pearson_diff_acc = 0

        self.rkl_list_1 = []
        self.rkl_fair_list_1 = []
        self.rkl_baseline_list_1 = []
        self.rrd_list_1 = []
        self.rrd_fair_list_1 = []
        self.rnd_list_1 = []
        self.rnd_fair_list_1 = []
        self.rkl_acc_1 = 0
        self.rkl_fair_acc_1 = 0
        self.rkl_baseline_acc_1 = 0
        self.rnd_acc_1 = 0
        self.rnd_fair_acc_1 = 0
        self.rrd_acc_1 = 0
        self.rrd_fair_acc_1 = 0

        self.rkl_list_2 = []
        self.rkl_fair_list_2 = []
        self.rkl_baseline_list_2 = []
        self.rrd_list_2 = []
        self.rrd_fair_list_2 = []
        self.rnd_list_2 = []
        self.rnd_fair_list_2 = []
        self.rkl_acc_2 = 0
        self.rkl_fair_acc_2 = 0
        self.rkl_baseline_acc_2 = 0
        self.rnd_acc_2 = 0
        self.rnd_fair_acc_2 = 0
        self.rrd_acc_2 = 0
        self.rrd_fair_acc_2 = 0

        self.rkl_list_5 = []
        self.rkl_fair_list_5 = []
        self.rkl_baseline_list_5 = []
        self.rrd_list_5 = []
        self.rrd_fair_list_5 = []
        self.rnd_list_5 = []
        self.rnd_fair_list_5 = []
        self.rkl_acc_5 = 0
        self.rkl_fair_acc_5 = 0
        self.rkl_baseline_acc_5 = 0
        self.rnd_acc_5 = 0
        self.rnd_fair_acc_5 = 0
        self.rrd_acc_5 = 0
        self.rrd_fair_acc_5 = 0

        self.rkl_list_10 = []
        self.rkl_fair_list_10 = []
        self.rkl_baseline_list_10 = []
        self.rrd_list_10 = []
        self.rrd_fair_list_10 = []
        self.rnd_list_10 = []
        self.rnd_fair_list_10 = []
        self.rkl_acc_10 = 0
        self.rkl_fair_acc_10 = 0
        self.rkl_baseline_acc_10 = 0
        self.rnd_acc_10 = 0
        self.rnd_fair_acc_10 = 0
        self.rrd_acc_10 = 0
        self.rrd_fair_acc_10 = 0

        self.premium_winners = 0
        self.non_premium_winners = 0
        self.fair_premium_winners = 0
        self.fair_non_premium_winners = 0
        self.baseline_premium_winners = 0
        self.baseline_non_premium_winners = 0
        self.prices_list = []
        self.delivery_time_list = []
        self.prices_fair_list = []
        self.delivery_time_fair_list = []
        self.prices_baseline_list = []
        self.delivery_time_baseline_list = []
        self.log_dataframe = pd.DataFrame(columns=[
            'premium_ratio',
            'premium_seller_num',
            'mean_price',
            'std_price',
            'mean_delivery_time',
            'std_delivery_time',
            'mean_fair_price',
            'std_fair_price',
            'mean_fair_delivery_time',
            'std_fair_delivery_time',
            'mean_rkl_1',
            'std_rkl_1',
            'mean_fair_rkl_1',
            'std_fair_rkl_1',
            'mean_rkl_2',
            'std_rkl_2',
            'mean_fair_rkl_2',
            'std_fair_rkl_2',
            'mean_rkl_5',
            'std_rkl_5',
            'mean_fair_rkl_5',
            'std_fair_rkl_5',
            'mean_rkl_10',
            'std_rkl_10',
            'mean_fair_rkl_10',
            'std_fair_rkl_10',
            'mean_rrd_1',
            'std_rrd_1',
            'mean_fair_rrd_1',
            'std_fair_rrd_1',
            'mean_rrd_2',
            'std_rrd_2',
            'mean_fair_rrd_2',
            'std_fair_rrd_2',
            'mean_rrd_5',
            'std_rrd_5',
            'mean_fair_rrd_5',
            'std_fair_rrd_5',
            'mean_rrd_10',
            'std_rrd_10',
            'mean_fair_rrd_10',
            'std_fair_rrd_10',
            'mean_rnd_1',
            'std_rnd_1',
            'mean_fair_rnd_1',
            'std_fair_rnd_1',
            'mean_rnd_2',
            'std_rnd_2',
            'mean_fair_rnd_2',
            'std_fair_rnd_2',
            'mean_rnd_5',
            'std_rnd_5',
            'mean_fair_rnd_5',
            'std_fair_rnd_5',
            'mean_rnd_10',
            'std_rnd_10',
            'mean_fair_rnd_10',
            'std_fair_rnd_10',
            'mean_position_dist',
            'mean_kendall_dist',
            'mean_spearman_dist',
            'mean_score_diff',
            'mean_pearson_diff',
            'mean_baseline_price',
            'std_baseline_price',
            'mean_baseline_delivery_time',
            'std_baseline_delivery_time',
            'mean_baseline_rkl_1',
            'std_baseline_rkl_1',
            'mean_baseline_rkl_2',
            'std_baseline_rkl_2',
            'mean_baseline_rkl_5',
            'std_baseline_rkl_5',
            'mean_baseline_rkl_10',
            'std_baseline_rkl_10',
            'premium_winners',
            'non_premium_winners',
            'fair_premium_winners',
            'fair_non_premium_winners',
            'baseline_premium_winners',
            'baseline_non_premium_winners',
        ])

    def generate_init(self):
        # generating data
        products_list = self.product_select.sample(self.size)
        prices_list = self.price_sampler.sample(products_list)
        delivery_time_list = self.delivery_time_sampler.sample(products_list)
        seller_score_list = self.seller_score_sampler.sample(products_list)
        installment_ratio_list = self.installment_sampler.sample(products_list)

        premium_prices_list = self.price_sampler.sample(products_list)
        premium_seller_score_list = self.seller_score_sampler.sample(products_list)
        premium_installment_list = self.installment_sampler.sample(products_list)
        premium_delivery_time_list = self.delivery_time_sampler.sample(products_list)

        # generating premium sellers
        self.seller_dict = {}
        for i in range(self.premium_size):
            name = name_generator()
            self.seller_dict[name] = Seller(
                name=name,
                price=premium_prices_list[i],
                premium=True,
                cluster=products_list[i],
                delivery_time=premium_delivery_time_list[i],
                seller_score=premium_seller_score_list[i],
                installment_ratio=premium_installment_list[i]
            )
        for i in range(self.premium_size, self.size):
            name = name_generator()
            self.seller_dict[name] = Seller(
                name=name,
                price=prices_list[i],
                premium=False,
                cluster=products_list[i],
                delivery_time=delivery_time_list[i],
                seller_score=seller_score_list[i],
                installment_ratio=installment_ratio_list[i]
            )

    def generate(self):
        for i in range(len(self.prices_list)):
            pprice = pow((self.prices_list[i] - (self.price_acc / self.reset_each)), 2)
            pdelivery = pow((self.delivery_time_list[i] - (self.delivery_time_acc / self.reset_each)), 2)

            prkl_1 = pow((self.rkl_list_1[i] - (self.rkl_acc_1 / self.reset_each)), 2)
            prkl_fair_1 = pow((self.rkl_fair_list_1[i] - (self.rkl_fair_acc_1 / self.reset_each)), 2)
            prkl_baseline_1 = pow((self.rkl_baseline_list_1[i] - (self.rkl_baseline_acc_1 / self.reset_each)), 2)

            prrd_1 = pow((self.rrd_list_1[i] - (self.rrd_acc_1 / self.reset_each)), 2)
            prrd_fair_1 = pow((self.rrd_fair_list_1[i] - (self.rrd_fair_acc_1 / self.reset_each)), 2)

            prnd_1 = pow((self.rnd_list_1[i] - (self.rnd_acc_1 / self.reset_each)), 2)
            prnd_fair_1 = pow((self.rnd_fair_list_1[i] - (self.rnd_fair_acc_1 / self.reset_each)), 2)

            prkl_2 = pow((self.rkl_list_2[i] - (self.rkl_acc_2 / self.reset_each)), 2)
            prkl_fair_2 = pow((self.rkl_fair_list_2[i] - (self.rkl_fair_acc_2 / self.reset_each)), 2)
            prkl_baseline_2 = pow((self.rkl_baseline_list_2[i] - (self.rkl_baseline_acc_2 / self.reset_each)), 2)

            prrd_2 = pow((self.rrd_list_2[i] - (self.rrd_acc_2 / self.reset_each)), 2)
            prrd_fair_2 = pow((self.rrd_fair_list_2[i] - (self.rrd_fair_acc_2 / self.reset_each)), 2)

            prnd_2 = pow((self.rnd_list_2[i] - (self.rnd_acc_2 / self.reset_each)), 2)
            prnd_fair_2 = pow((self.rnd_fair_list_2[i] - (self.rnd_fair_acc_2 / self.reset_each)), 2)

            prkl_10 = pow((self.rkl_list_10[i] - (self.rkl_acc_10 / self.reset_each)), 2)
            prkl_fair_10 = pow((self.rkl_fair_list_10[i] - (self.rkl_fair_acc_10 / self.reset_each)), 2)
            prkl_baseline_10 = pow((self.rkl_baseline_list_10[i] - (self.rkl_baseline_acc_10 / self.reset_each)), 2)

            prrd_10 = pow((self.rrd_list_10[i] - (self.rrd_acc_10 / self.reset_each)), 2)
            prrd_fair_10 = pow((self.rrd_fair_list_10[i] - (self.rrd_fair_acc_10 / self.reset_each)), 2)

            prnd_10 = pow((self.rnd_list_10[i] - (self.rnd_acc_10 / self.reset_each)), 2)
            prnd_fair_10 = pow((self.rnd_fair_list_10[i] - (self.rnd_fair_acc_10 / self.reset_each)), 2)

            prkl_5 = pow((self.rkl_list_5[i] - (self.rkl_acc_5 / self.reset_each)), 2)
            prkl_fair_5 = pow((self.rkl_fair_list_5[i] - (self.rkl_fair_acc_5 / self.reset_each)), 2)
            prkl_baseline_5 = pow((self.rkl_baseline_list_5[i] - (self.rkl_baseline_acc_5 / self.reset_each)), 2)

            prrd_5 = pow((self.rrd_list_5[i] - (self.rrd_acc_5 / self.reset_each)), 2)
            prrd_fair_5 = pow((self.rrd_fair_list_5[i] - (self.rrd_fair_acc_5 / self.reset_each)), 2)

            prnd_5 = pow((self.rnd_list_5[i] - (self.rnd_acc_5 / self.reset_each)), 2)
            prnd_fair_5 = pow((self.rnd_fair_list_5[i] - (self.rnd_fair_acc_5 / self.reset_each)), 2)

            pprice_fair = pow((self.delivery_time_fair_list[i] - (self.delivery_time_fair_acc / self.reset_each)), 2)
            pdelivery_fair = pow((self.prices_fair_list[i] - (self.price_fair_acc / self.reset_each)), 2)

            pprice_baseline = pow((self.delivery_time_baseline_list[i] - (self.delivery_time_baseline_acc / self.reset_each)), 2)
            pdelivery_baseline = pow((self.prices_baseline_list[i] - (self.price_baseline_acc / self.reset_each)), 2)


        log_row = [
            self.premium_ratio,
            self.premium_size,
            self.price_acc / self.reset_each,
            sqrt(pprice/self.reset_each),
            self.delivery_time_acc / self.reset_each,
            sqrt(pdelivery / self.reset_each),
            self.price_fair_acc / self.reset_each,
            sqrt(pprice_fair / self.reset_each),
            self.delivery_time_fair_acc / self.reset_each,
            sqrt(pdelivery_fair / self.reset_each),
            self.rkl_acc_1 / self.reset_each,
            sqrt(prkl_1 / self.reset_each),
            self.rkl_fair_acc_1 / self.reset_each,
            sqrt(prkl_fair_1 / self.reset_each),
            self.rkl_acc_2 / self.reset_each,
            sqrt(prkl_2 / self.reset_each),
            self.rkl_fair_acc_2 / self.reset_each,
            sqrt(prkl_fair_2 / self.reset_each),
            self.rkl_acc_5 / self.reset_each,
            sqrt(prkl_5 / self.reset_each),
            self.rkl_fair_acc_5 / self.reset_each,
            sqrt(prkl_fair_5 / self.reset_each),
            self.rkl_acc_10 / self.reset_each,
            sqrt(prkl_10 / self.reset_each),
            self.rkl_fair_acc_10 / self.reset_each,
            sqrt(prkl_fair_10 / self.reset_each),
            self.rrd_acc_1 / self.reset_each,
            sqrt(prrd_1 / self.reset_each),
            self.rrd_fair_acc_1 / self.reset_each,
            sqrt(prrd_fair_1 / self.reset_each),
            self.rrd_acc_2 / self.reset_each,
            sqrt(prrd_2 / self.reset_each),
            self.rrd_fair_acc_2 / self.reset_each,
            sqrt(prrd_fair_2 / self.reset_each),
            self.rrd_acc_5 / self.reset_each,
            sqrt(prrd_5 / self.reset_each),
            self.rrd_fair_acc_5 / self.reset_each,
            sqrt(prrd_fair_5 / self.reset_each),
            self.rrd_acc_10 / self.reset_each,
            sqrt(prrd_10 / self.reset_each),
            self.rrd_fair_acc_10 / self.reset_each,
            sqrt(prrd_fair_10 / self.reset_each),
            self.rnd_acc_1 / self.reset_each,
            sqrt(prnd_1 / self.reset_each),
            self.rnd_fair_acc_1 / self.reset_each,
            sqrt(prnd_fair_1 / self.reset_each),
            self.rnd_acc_2 / self.reset_each,
            sqrt(prnd_2 / self.reset_each),
            self.rnd_fair_acc_2 / self.reset_each,
            sqrt(prnd_fair_2 / self.reset_each),
            self.rnd_acc_5 / self.reset_each,
            sqrt(prnd_5 / self.reset_each),
            self.rnd_fair_acc_5 / self.reset_each,
            sqrt(prnd_fair_5 / self.reset_each),
            self.rnd_acc_10 / self.reset_each,
            sqrt(prnd_10 / self.reset_each),
            self.rnd_fair_acc_10 / self.reset_each,
            sqrt(prnd_fair_10 / self.reset_each),
            self.position_dist_acc / self.reset_each,
            self.kendall_dist_acc / self.reset_each,
            self.spearman_dist_acc / self.reset_each,
            self.score_diff_acc / self.reset_each,
            self.pearson_diff_acc / self.reset_each,
            self.price_baseline_acc / self.reset_each,
            sqrt(pprice_baseline / self.reset_each),
            self.delivery_time_baseline_acc / self.reset_each,
            sqrt(pdelivery_baseline / self.reset_each),
            self.rkl_baseline_acc_1 / self.reset_each,
            sqrt(prkl_baseline_1 / self.reset_each),
            self.rkl_baseline_acc_2 / self.reset_each,
            sqrt(prkl_baseline_2 / self.reset_each),
            self.rkl_baseline_acc_5 / self.reset_each,
            sqrt(prkl_baseline_5 / self.reset_each),
            self.rkl_baseline_acc_10 / self.reset_each,
            sqrt(prkl_baseline_10 / self.reset_each),
            self.premium_winners,
            self.non_premium_winners,
            self.fair_premium_winners,
            self.fair_non_premium_winners,
            self.baseline_premium_winners,
            self.baseline_non_premium_winners,
        ]
        self.log_dataframe.loc[len(self.log_dataframe)] = log_row
        self.price_acc = 0
        self.premium_winners = 0
        self.non_premium_winners = 0
        self.fair_premium_winners = 0
        self.fair_non_premium_winners = 0
        self.baseline_premium_winners = 0
        self.baseline_non_premium_winners = 0
        self.price_fair_acc = 0
        self.delivery_time_fair_acc = 0
        self.delivery_time_acc = 0
        self.price_baseline_acc = 0
        self.delivery_time_baseline_acc = 0
        self.prices_list = []

        self.position_dist_acc = 0
        self.kendall_dist_acc = 0
        self.spearman_dist_acc = 0
        self.score_diff_acc = 0
        self.pearson_diff_acc = 0

        self.rkl_list_1 = []
        self.rkl_fair_list_1 = []
        self.rkl_baseline_list_1 = []
        self.rrd_list_1 = []
        self.rrd_fair_list_1 = []
        self.rnd_list_1 = []
        self.rnd_fair_list_1 = []
        self.rkl_acc_1 = 0
        self.rkl_fair_acc_1 = 0
        self.rkl_baseline_acc_1 = 0
        self.rnd_acc_1 = 0
        self.rnd_fair_acc_1 = 0
        self.rrd_acc_1 = 0
        self.rrd_fair_acc_1 = 0

        self.rkl_list_2 = []
        self.rkl_fair_list_2 = []
        self.rkl_baseline_list_2 = []
        self.rrd_list_2 = []
        self.rrd_fair_list_2 = []
        self.rnd_list_2 = []
        self.rnd_fair_list_2 = []
        self.rkl_acc_2 = 0
        self.rkl_fair_acc_2 = 0
        self.rkl_baseline_acc_2 = 0
        self.rnd_acc_2 = 0
        self.rnd_fair_acc_2 = 0
        self.rrd_acc_2 = 0
        self.rrd_fair_acc_2 = 0

        self.rkl_list_5 = []
        self.rkl_fair_list_5 = []
        self.rkl_baseline_list_5 = []
        self.rrd_list_5 = []
        self.rrd_fair_list_5 = []
        self.rnd_list_5 = []
        self.rnd_fair_list_5 = []
        self.rkl_acc_5 = 0
        self.rkl_fair_acc_5 = 0
        self.rkl_baseline_acc_5 = 0
        self.rnd_acc_5 = 0
        self.rnd_fair_acc_5 = 0
        self.rrd_acc_5 = 0
        self.rrd_fair_acc_5 = 0

        self.rkl_list_10 = []
        self.rkl_fair_list_10 = []
        self.rkl_baseline_list_10 = []
        self.rrd_list_10 = []
        self.rrd_fair_list_10 = []
        self.rnd_list_10 = []
        self.rnd_fair_list_10 = []
        self.rkl_acc_10 = 0
        self.rkl_fair_acc_10 = 0
        self.rkl_baseline_acc_10 = 0
        self.rnd_acc_10 = 0
        self.rnd_fair_acc_10 = 0
        self.rrd_acc_10 = 0
        self.rrd_fair_acc_10 = 0

        self.delivery_time_list = []
        self.prices_fair_list = []
        self.delivery_time_fair_list = []
        self.prices_baseline_list = []
        self.delivery_time_baseline_list = []
        # tradeoff
        # seller_list = list(self.seller_dict.values())
        # seller_list.sort(key=lambda x: x.mean_position, reverse=False)
        # for i in range(self.premium_size):
        #     seller_list[i].premium = not seller_list[i].premium

        self.premium_size += self.tradeoff_ratio

        products_list = self.product_select.sample(self.size)
        prices_list = self.price_sampler.sample(products_list)
        delivery_time_list = self.delivery_time_sampler.sample(products_list)
        seller_score_list = self.seller_score_sampler.sample(products_list)
        installment_ratio_list = self.installment_sampler.sample(products_list)

        premium_prices_list = self.price_sampler.sample(products_list)
        premium_seller_score_list = self.seller_score_sampler.sample(products_list)
        premium_installment_list = self.installment_sampler.sample(products_list)
        premium_delivery_time_list = self.delivery_time_sampler.sample(products_list)


        self.seller_dict = {}

        for i in range(self.premium_size):
            name = name_generator()
            self.seller_dict[name] = Seller(
                name=name,
                price=premium_prices_list[i],
                premium=True,
                cluster=products_list[i],
                delivery_time=premium_delivery_time_list[i],
                seller_score=premium_seller_score_list[i],
                installment_ratio=premium_installment_list[i]
            )
        for i in range(self.premium_size, self.size):
            name = name_generator()
            self.seller_dict[name] = Seller(
                name=name,
                price=prices_list[i],
                premium=False,
                cluster=products_list[i],
                delivery_time=delivery_time_list[i],
                seller_score=seller_score_list[i],
                installment_ratio=installment_ratio_list[i]
            )
        self.premium_ratio = (self.premium_size * 100 / self.size) / 100

    def generate_sample(self, size: int) -> np.array:
        product_type = self.product_select.sample(1)
        product_type = product_type[0]
        seller_product_type_list = [seller
                                    for seller in list(self.seller_dict.values())
                                    if seller.cluster == product_type]
        seller_list = random.choices(seller_product_type_list, k=size)
        valid_flag = False
        while not valid_flag:
            count_premium = 0
            count_non_premium = 0
            for s in seller_list:
                if s.premium:
                    count_premium += 1
                else:
                    count_non_premium += 1
            if not count_premium > 0 or not count_non_premium > 0:
                seller_list = random.choices(seller_product_type_list, k=size)
            else:
                valid_flag = True
        return seller_list




class MarketplaceSimulation:
    def __init__(self,
                 seller_pool: SellerPool,
                 num_iterations: int,
                 num_offers: int,
                 reset_each: int,
                 agent: Agent,
                 tradeoff_rank: str,
                 fair_agent: Agent,
                 logger: LoggerData):

        # attributes
        self.seller_pool = seller_pool
        self.num_iterations = num_iterations
        self.reset_each = reset_each
        self.agent = agent
        self.tradeoff_rank = tradeoff_rank
        self.fair_agent = fair_agent
        self.num_offers = num_offers
        self.logger = logger
        self.reset_count = 0
        self.iteration_count = 0

    def _it_count(self):
        self.iteration_count += 1

    def _step(self):
        # reset if applicable
        self.reset_count += 1
        if self.reset_count == self.reset_each:
            self.reset_count = 0
            self.seller_pool.generate()

        # sample data
        data = []
        seller_list = self.seller_pool.generate_sample(self.num_offers)
        for seller in seller_list:
            data.append(seller.ordered_list)
        data = pd.DataFrame(data, columns=COLUMNS)
        data = self.agent.rank(data)
        data = self.fair_agent.rank(data)

        position_fair_list = data['fair_rank'].to_list()
        position_list = data['rank'].to_list()
        position_baseline_list = data['rank_baseline'].to_list()

        prices = data['price'].to_list()
        delivery_times = data['delivery_time'].to_list()
        premium_list = data['premium'].to_list()

        non_premium_positions = []
        non_premium_positions_baseline = []
        non_premium_positions_fair = []
        ranking_fair = [0] * self.num_offers
        ranking_unfair = [0] * self.num_offers
        ranking_baseline = [0] * self.num_offers

        for i in range(len(premium_list)):
            ranking_unfair[position_list[i] - 1] = i
            ranking_fair[position_fair_list[i] - 1] = i
            ranking_baseline[position_baseline_list[i] - 1] = i
            if premium_list[i] == False:
                non_premium_positions.append(i)
                non_premium_positions_fair.append(i)
                non_premium_positions_baseline.append(i)

        normalizer = getNormalizer(
            _user_N=len(ranking_unfair),
            _pro_N=len(non_premium_positions),
            _gf_measure='rKL')

        self.seller_pool.score_diff_acc += calculateScoreDifference(sorted(data['scores'].to_list()), sorted(data['fair_scores'].to_list()))

        self.seller_pool.position_dist_acc += calculatePositionDifference(ranking_fair, ranking_unfair)

        self.seller_pool.kendall_dist_acc += calculateKendallDistance(ranking_fair, ranking_unfair)

        self.seller_pool.spearman_dist_acc += spearmanr(data['scores'].to_list(),data['fair_scores'].to_list())[0]

        self.seller_pool.pearson_diff_acc += pearsonr(data['scores'].to_list(), data['fair_scores'].to_list())[0]

        rkl_1 = calculateNDFairness(_ranking=ranking_unfair,
                                  _cut_point=1,
                                  _protected_group=non_premium_positions,
                                  _gf_measure='rKL',
                                  _normalizer=normalizer
                                  )
        
        rkl_fair_1 = calculateNDFairness(_ranking=ranking_fair,
                                  _cut_point=1,
                                  _protected_group=non_premium_positions_fair,
                                  _gf_measure='rKL',
                                  _normalizer=normalizer
                                  )

        rkl_baseline_1 = calculateNDFairness(_ranking=ranking_baseline,
                                  _cut_point=1,
                                  _protected_group=non_premium_positions_baseline,
                                  _gf_measure='rKL',
                                  _normalizer=normalizer
                                  )

        rnd_1 = calculateNDFairness(_ranking=ranking_unfair,
                                  _cut_point=1,
                                  _protected_group=non_premium_positions,
                                  _gf_measure='rND',
                                  _normalizer=normalizer
                                  )

        rnd_fair_1 = calculateNDFairness(_ranking=ranking_fair,
                                       _cut_point=1,
                                       _protected_group=non_premium_positions_fair,
                                       _gf_measure='rND',
                                       _normalizer=normalizer
                                       )

        rrd_1 = calculateNDFairness(_ranking=ranking_unfair,
                                  _cut_point=1,
                                  _protected_group=non_premium_positions,
                                  _gf_measure='rRD',
                                  _normalizer=normalizer
                                  )

        rrd_fair_1 = calculateNDFairness(_ranking=ranking_fair,
                                       _cut_point=1,
                                       _protected_group=non_premium_positions_fair,
                                       _gf_measure='rRD',
                                       _normalizer=normalizer
                                       )

        rkl_2 = calculateNDFairness(_ranking=ranking_unfair,
                                    _cut_point=2,
                                    _protected_group=non_premium_positions,
                                    _gf_measure='rKL',
                                    _normalizer=normalizer
                                    )

        rkl_fair_2 = calculateNDFairness(_ranking=ranking_fair,
                                         _cut_point=2,
                                         _protected_group=non_premium_positions_fair,
                                         _gf_measure='rKL',
                                         _normalizer=normalizer
                                         )

        rkl_baseline_2 = calculateNDFairness(_ranking=ranking_baseline,
                                  _cut_point=2,
                                  _protected_group=non_premium_positions_baseline,
                                  _gf_measure='rKL',
                                  _normalizer=normalizer
                                  )

        rnd_2 = calculateNDFairness(_ranking=ranking_unfair,
                                    _cut_point=2,
                                    _protected_group=non_premium_positions,
                                    _gf_measure='rND',
                                    _normalizer=normalizer
                                    )

        rnd_fair_2 = calculateNDFairness(_ranking=ranking_fair,
                                         _cut_point=2,
                                         _protected_group=non_premium_positions_fair,
                                         _gf_measure='rND',
                                         _normalizer=normalizer
                                         )

        rrd_2 = calculateNDFairness(_ranking=ranking_unfair,
                                    _cut_point=2,
                                    _protected_group=non_premium_positions,
                                    _gf_measure='rRD',
                                    _normalizer=normalizer
                                    )

        rrd_fair_2 = calculateNDFairness(_ranking=ranking_fair,
                                         _cut_point=2,
                                         _protected_group=non_premium_positions_fair,
                                         _gf_measure='rRD',
                                         _normalizer=normalizer
                                         )

        rkl_5 = calculateNDFairness(_ranking=ranking_unfair,
                                    _cut_point=5,
                                    _protected_group=non_premium_positions,
                                    _gf_measure='rKL',
                                    _normalizer=normalizer
                                    )

        rkl_fair_5 = calculateNDFairness(_ranking=ranking_fair,
                                         _cut_point=5,
                                         _protected_group=non_premium_positions_fair,
                                         _gf_measure='rKL',
                                         _normalizer=normalizer
                                         )

        rkl_baseline_5 = calculateNDFairness(_ranking=ranking_baseline,
                                  _cut_point=5,
                                  _protected_group=non_premium_positions_baseline,
                                  _gf_measure='rKL',
                                  _normalizer=normalizer
                                  )

        rnd_5 = calculateNDFairness(_ranking=ranking_unfair,
                                    _cut_point=5,
                                    _protected_group=non_premium_positions,
                                    _gf_measure='rND',
                                    _normalizer=normalizer
                                    )

        rnd_fair_5 = calculateNDFairness(_ranking=ranking_fair,
                                         _cut_point=5,
                                         _protected_group=non_premium_positions_fair,
                                         _gf_measure='rND',
                                         _normalizer=normalizer
                                         )

        rrd_5 = calculateNDFairness(_ranking=ranking_unfair,
                                    _cut_point=5,
                                    _protected_group=non_premium_positions,
                                    _gf_measure='rRD',
                                    _normalizer=normalizer
                                    )

        rrd_fair_5 = calculateNDFairness(_ranking=ranking_fair,
                                         _cut_point=5,
                                         _protected_group=non_premium_positions_fair,
                                         _gf_measure='rRD',
                                         _normalizer=normalizer
                                         )

        rkl_10 = calculateNDFairness(_ranking=ranking_unfair,
                                    _cut_point=10,
                                    _protected_group=non_premium_positions,
                                    _gf_measure='rKL',
                                    _normalizer=normalizer
                                    )

        rkl_fair_10 = calculateNDFairness(_ranking=ranking_fair,
                                         _cut_point=10,
                                         _protected_group=non_premium_positions_fair,
                                         _gf_measure='rKL',
                                         _normalizer=normalizer
                                         )

        rkl_baseline_10 = calculateNDFairness(_ranking=ranking_baseline,
                                  _cut_point=10,
                                  _protected_group=non_premium_positions_baseline,
                                  _gf_measure='rKL',
                                  _normalizer=normalizer
                                  )

        rnd_10 = calculateNDFairness(_ranking=ranking_unfair,
                                    _cut_point=10,
                                    _protected_group=non_premium_positions,
                                    _gf_measure='rND',
                                    _normalizer=normalizer
                                    )

        rnd_fair_10 = calculateNDFairness(_ranking=ranking_fair,
                                         _cut_point=10,
                                         _protected_group=non_premium_positions_fair,
                                         _gf_measure='rND',
                                         _normalizer=normalizer
                                         )

        rrd_10 = calculateNDFairness(_ranking=ranking_unfair,
                                    _cut_point=10,
                                    _protected_group=non_premium_positions,
                                    _gf_measure='rRD',
                                    _normalizer=normalizer
                                    )

        rrd_fair_10 = calculateNDFairness(_ranking=ranking_fair,
                                         _cut_point=10,
                                         _protected_group=non_premium_positions_fair,
                                         _gf_measure='rRD',
                                         _normalizer=normalizer
                                         )

        index_winner = position_list.index(1)
        index_winner_fair = position_fair_list.index(1)
        index_winner_baseline = position_baseline_list.index(1)
        self.seller_pool.price_acc += prices[index_winner]
        self.seller_pool.prices_list.append(prices[index_winner])
        self.seller_pool.price_baseline_acc += prices[index_winner_baseline]
        self.seller_pool.prices_baseline_list.append(prices[index_winner_baseline])
        self.seller_pool.price_fair_acc += prices[index_winner_fair]
        self.seller_pool.prices_fair_list.append(prices[index_winner_fair])
        self.seller_pool.delivery_time_acc += delivery_times[index_winner]
        self.seller_pool.delivery_time_list.append(delivery_times[index_winner])
        self.seller_pool.delivery_time_baseline_acc += delivery_times[index_winner_baseline]
        self.seller_pool.delivery_time_baseline_list.append(delivery_times[index_winner_baseline])
        self.seller_pool.delivery_time_fair_acc += delivery_times[index_winner_fair]
        self.seller_pool.delivery_time_fair_list.append(delivery_times[index_winner_fair])

        self.seller_pool.rkl_fair_acc_1 += rkl_fair_1
        self.seller_pool.rkl_acc_1 += rkl_1
        self.seller_pool.rkl_baseline_acc_1 += rkl_baseline_1
        self.seller_pool.rkl_list_1.append(rkl_1)
        self.seller_pool.rkl_fair_list_1.append(rkl_fair_1)
        self.seller_pool.rkl_baseline_list_1.append(rkl_baseline_1)

        self.seller_pool.rrd_fair_acc_1 += rrd_fair_1
        self.seller_pool.rrd_acc_1 += rrd_1
        self.seller_pool.rrd_list_1.append(rrd_1)
        self.seller_pool.rrd_fair_list_1.append(rrd_fair_1)

        self.seller_pool.rnd_fair_acc_1 += rnd_fair_1
        self.seller_pool.rnd_acc_1 += rnd_1
        self.seller_pool.rnd_list_1.append(rnd_1)
        self.seller_pool.rnd_fair_list_1.append(rnd_fair_1)

        self.seller_pool.rkl_fair_acc_2 += rkl_fair_2
        self.seller_pool.rkl_acc_2 += rkl_2
        self.seller_pool.rkl_baseline_acc_2 += rkl_baseline_2
        self.seller_pool.rkl_list_2.append(rkl_2)
        self.seller_pool.rkl_fair_list_2.append(rkl_fair_2)
        self.seller_pool.rkl_baseline_list_2.append(rkl_baseline_2)

        self.seller_pool.rrd_fair_acc_2 += rrd_fair_2
        self.seller_pool.rrd_acc_2 += rrd_2
        self.seller_pool.rrd_list_2.append(rrd_2)
        self.seller_pool.rrd_fair_list_2.append(rrd_fair_2)

        self.seller_pool.rnd_fair_acc_2 += rnd_fair_2
        self.seller_pool.rnd_acc_2 += rnd_2
        self.seller_pool.rnd_list_2.append(rnd_2)
        self.seller_pool.rnd_fair_list_2.append(rnd_fair_2)

        self.seller_pool.rkl_fair_acc_10 += rkl_fair_10
        self.seller_pool.rkl_acc_10 += rkl_10
        self.seller_pool.rkl_baseline_acc_10 += rkl_baseline_10
        self.seller_pool.rkl_list_10.append(rkl_10)
        self.seller_pool.rkl_fair_list_10.append(rkl_fair_10)
        self.seller_pool.rkl_baseline_list_10.append(rkl_baseline_10)

        self.seller_pool.rrd_fair_acc_10 += rrd_fair_10
        self.seller_pool.rrd_acc_10 += rrd_10
        self.seller_pool.rrd_list_10.append(rrd_10)
        self.seller_pool.rrd_fair_list_10.append(rrd_fair_10)

        self.seller_pool.rnd_fair_acc_10 += rnd_fair_10
        self.seller_pool.rnd_acc_10 += rnd_10
        self.seller_pool.rnd_list_10.append(rnd_10)
        self.seller_pool.rnd_fair_list_10.append(rnd_fair_10)

        self.seller_pool.rkl_fair_acc_5 += rkl_fair_5
        self.seller_pool.rkl_acc_5 += rkl_5
        self.seller_pool.rkl_baseline_acc_5 += rkl_baseline_5
        self.seller_pool.rkl_list_5.append(rkl_5)
        self.seller_pool.rkl_fair_list_5.append(rkl_fair_5)
        self.seller_pool.rkl_baseline_list_5.append(rkl_baseline_5)

        self.seller_pool.rrd_fair_acc_5 += rrd_fair_5
        self.seller_pool.rrd_acc_5 += rrd_5
        self.seller_pool.rrd_list_5.append(rrd_5)
        self.seller_pool.rrd_fair_list_5.append(rrd_fair_5)

        self.seller_pool.rnd_fair_acc_5 += rnd_fair_5
        self.seller_pool.rnd_acc_5 += rnd_5
        self.seller_pool.rnd_list_5.append(rnd_5)
        self.seller_pool.rnd_fair_list_5.append(rnd_fair_5)

        if premium_list[index_winner]:
            self.seller_pool.premium_winners += 1
        else:
            self.seller_pool.non_premium_winners += 1

        if premium_list[index_winner_fair]:
            self.seller_pool.fair_premium_winners += 1
        else:
            self.seller_pool.fair_non_premium_winners += 1

        if premium_list[index_winner_baseline]:
            self.seller_pool.baseline_premium_winners += 1
        else:
            self.seller_pool.baseline_non_premium_winners += 1

        # update sellers
        for i in range(len(seller_list)):
            seller_i = seller_list[i]
            seller_i.participations += 1
            if self.tradeoff_rank == 'fair':
                seller_i.position_acc += position_fair_list[i]
            else:
                seller_i.position_acc += position_list[i]

            self.seller_pool.seller_dict[seller_i.name] = seller_i

        self.logger.log_data(data, self.iteration_count)

    def run_simulation(self):
        self.seller_pool.generate_init()
        while self.iteration_count <= self.num_iterations:
            print("Iteraction {}/{}".format(self.iteration_count, self.num_iterations))
            print("Premium ratio: {}".format(self.seller_pool.premium_ratio))
            self._step()
            self._it_count()
        self.logger.data['scores'] = minmax_scale(self.logger.data['scores'], (0.1, 1))
        self.logger.data['fair_scores'] = minmax_scale(self.logger.data['fair_scores'], (0.1, 1))
        self.logger.save()
