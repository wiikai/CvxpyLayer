import os
import sys
sys.path.append(os.chdir('/home/rice/huangweikai'))

import numpy as np
import pandas as pd
import torch
from model import rolling_train
import quool
import random

cvlayer = quool.Factor("./data/cv-layer-factor", code_level="order_book_id", date_level="date")
cv_price = quool.Factor("./data/cv-layer-price", code_level="order_book_id", date_level="date")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def main():
    set_seed(0)
    # processor= [(madoutlier,{'dev': 5, 'drop': False}), zscore]
    stop = '20240701'
    rollback = cvlayer.get_trading_days_rollback(stop, 1800 + 63*9 + 21*2 + 30-2)

    # # FactorModel
    # data = cvlayer.read('1d_ret, 2d_ret, 3d_ret, 4d_ret, 5d_ret, 10d_ret, 10d_std, 20d_ret, 20d_std, 30d_ret, 30d_std',start=rollback, stop=stop, processor=None).swaplevel('date', 'order_book_id').sort_index()
    # future = cvlayer.read('20d_future_ret', start=rollback, stop=stop).stack().to_frame(name='future')
    # cov_matrix = data['1d_ret'].unstack().rolling(30).cov().dropna(how='all')
    # cov_matrix.columns.name = ''
    # data = pd.concat([data, cov_matrix, future], axis=1).loc[cov_matrix.index]

    # LSTMModel
    data = cvlayer.read('1d_ret, 20d_future_ret',start=rollback, stop=stop, processor=None).swaplevel('date', 'order_book_id').sort_index()
    data = data.rename(columns={'20d_future_ret':'future'})

    rolling_train(data, window_size=1800, train_size=1500, val_size=300, roll_step=63, gap=21, epochs=50, early_stopping=10)

if __name__ == '__main__':
    main()