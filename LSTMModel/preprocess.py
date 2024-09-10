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

class LSTMDataset(Dataset):
    def __init__(self, data, window_size=30):
        self.data = data
        self.window_size = window_size
        self.n_samples = len(data.index.get_level_values(0).unique()) - window_size + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        slice_data = self.data['1d_ret'].unstack().iloc[idx:idx + self.window_size].values
        cov_matrix = np.cov(slice_data.T)
        label = self.data['future'].unstack().iloc[idx + self.window_size - 1].values
        
        slice_data_tensor = torch.tensor(slice_data, dtype=torch.float32)
        cov_matrix_tensor = torch.tensor(cov_matrix, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return slice_data_tensor, cov_matrix_tensor, label_tensor
    

cvlayer = quool.Factor("./data/cv-layer-factor", code_level="order_book_id", date_level="date")
cv_price = quool.Factor("./data/cv-layer-price", code_level="order_book_id", date_level="date")


if __name__ == '__main__':
    processor= [(madoutlier,{'dev': 5, 'drop': False}), zscore]
    stop = '20240701'
    rollback = cvlayer.get_trading_days_rollback(stop, 1800 + 63*9 + 21*2 + 30-2)
    data = cvlayer.read('1d_ret, 20d_future_ret',start=rollback, stop=stop, processor=None).swaplevel('date', 'order_book_id').sort_index()
    data = data.rename(columns={'20d_future_ret':'future'})
    data.to_parquet('test.parquet')