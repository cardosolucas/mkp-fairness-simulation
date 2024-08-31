import numpy as np
import pandas as pd

class LoggerData:
    def __init__(self, path_generated_data: str):
        self.path_generated_data = path_generated_data
        self.data = None

    def log_data(self, data: pd.DataFrame, iteration: int):
        if self.data is None:
            self.data = data
            self.data['order'] = iteration
        else:
            data['order'] = iteration
            self.data = pd.concat([self.data, data], ignore_index=True)
            #self.data = self.data.reset_index()

    def save(self):
        self.data.to_csv(self.path_generated_data, index=False, header=True)

    # def get_incremental_data(self, lower_it: int) -> tuple:
    #     data_snp = self.data.query("order <= {}".format(lower_it))
    #     data_snp = data_snp[['price', 'delivery_time', 'seller_score', 'installment_ratio', 'premium', 'scores_mab']]
    #     data_train = data_snp.drop(['scores_mab'], axis=1)
    #     data_scores = data_snp['scores_mab']
    #     return data_train, data_scores