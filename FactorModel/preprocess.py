import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import quool

def cov_matrix_sqrt_svd(cov_matrix):
    U, S, Vt = np.linalg.svd(cov_matrix)
    return U @ np.diag(np.sqrt(S)) @ Vt

def minmax(df: pd.DataFrame):
    return df.sub(df.min(axis=1), axis=0).div(
        df.max(axis=1) - df.min(axis=1), axis=0)

def tensor_minmax(tensor):
    min_vals = tensor.min(dim=1, keepdim=True)[0]
    max_vals = tensor.max(dim=1, keepdim=True)[0]
    return (tensor - min_vals) / (max_vals - min_vals)

def zscore(df: pd.DataFrame):
    if isinstance(df.index, pd.MultiIndex):
        return df.groupby(level='date').transform(lambda x: (x - x.mean()) / x.std())
    else:
        return df.sub(df.mean(axis=1), axis=0
        ).div(df.std(axis=1), axis=0)

def madoutlier( 
    df: pd.DataFrame, 
    dev: int, 
    drop: bool = False
):
    def apply_mad(df: pd.DataFrame) -> pd.DataFrame:
        median = df.median(axis=1)
        ad = df.sub(median, axis=0)
        mad = ad.abs().median(axis=1)
        thresh_down = median - dev * mad
        thresh_up = median + dev * mad
        if not drop:
            return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
        return df.where(
            df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0),
            other=np.nan, axis=0).where(~df.isna())
    
    if isinstance(df.index, pd.MultiIndex):
        return df.apply(lambda x: apply_mad(x.unstack('order_book_id')).unstack())
    else:
        return apply_mad(df)

class FactorDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.time_slices = data.index.get_level_values(0).unique()
        self.cov_matrix = data.index.get_level_values(1).unique()
        self.label = 'future'

    def __len__(self):
        return len(self.time_slices)

    def __getitem__(self, idx):
        time_point = self.time_slices[idx]
        slice_data = self.data.loc[time_point]

        features = slice_data.drop(columns=[self.label] + list(self.cov_matrix)).values
        label = slice_data[self.label].values
        cov_matrix = slice_data[self.cov_matrix].values
        Q_sqrt = cov_matrix_sqrt_svd(cov_matrix)

        Q_sqrt_tensor = torch.tensor(Q_sqrt, dtype=torch.float32)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return features_tensor, Q_sqrt_tensor, label_tensor

cvlayer = quool.Factor("./data/cv-layer-factor", code_level="order_book_id", date_level="date")
cv_price = quool.Factor("./data/cv-layer-price", code_level="order_book_id", date_level="date")


if __name__ == '__main__':
    processor= [(madoutlier,{'dev': 5, 'drop': False}), zscore]
    stop = '20240701'
    rollback = cvlayer.get_trading_days_rollback(stop, 1800 + 63*9 + 21*2 + 30-2)

    data = cvlayer.read('1d_ret, 2d_ret, 3d_ret, 4d_ret, 5d_ret, 10d_ret, 10d_std, 20d_ret, 20d_std, 30d_ret, 30d_std',start=rollback, stop=stop, processor=None).swaplevel('date', 'order_book_id').sort_index()
    future = cvlayer.read('20d_future_ret', start=rollback, stop=stop).stack().to_frame(name='future')
    cov_matrix = data['1d_ret'].unstack().rolling(30).cov().dropna(how='all')
    cov_matrix.columns.name = ''
    data = pd.concat([data, cov_matrix, future], axis=1).loc[cov_matrix.index]
    data.to_parquet('test.parquet')