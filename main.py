from simulations import SellerPool, MarketplaceSimulation
from samplers import EmpiricalSampler
from agents import LinearAgent, FairAgent, FairAgentOptimizer
import joblib
from log import LoggerData


sampler_price = EmpiricalSampler(variable=0, premium=False)
sampler_delivery = EmpiricalSampler(variable=1, premium=False)
sampler_installment = EmpiricalSampler(variable=2, premium=False)
sampler_seller = EmpiricalSampler(variable=3, premium=False)

sampler_seller_premium = EmpiricalSampler(variable=3, premium=True)

seller_pool = SellerPool(
    size=1000,
    premium_ratio=0.1,
    tradeoff_ratio=30,
    price_sampler=sampler_price,
    reset_each=1000,
    delivery_time_sampler=sampler_delivery,
    seller_score_sampler=sampler_seller,
    installment_sampler=sampler_installment,
    premium_seller_score_sampler=sampler_seller_premium)

lin_agent = LinearAgent(weigths=[0.5, 0.2, 0.1, 0.1, 0.1])
fair_agent = FairAgentOptimizer()
logger = LoggerData('data/data_fair-v3-incr-100-score_new_rkl.csv')

simulator = MarketplaceSimulation(
    seller_pool=seller_pool,
    num_iterations=20000,
    num_offers=20,
    reset_each=1000,
    agent=lin_agent,
    tradeoff_rank='fair',
    fair_agent=fair_agent,
    logger=logger)

simulator.run_simulation()
simulator.seller_pool.log_dataframe.to_csv("data/tradeoff_data-v3-incr-100-score_new_rkl.csv")