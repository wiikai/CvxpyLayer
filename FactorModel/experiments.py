import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from preprocess import FactorDataset
from model import FactorModel
from torch.utils.data import DataLoader
import torch.optim as optim

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def train_model(train_dataloader, val_dataloader, model, optimizer, epochs=50, early_stopping=10):
    patience_counter = 0
    best_loss = np.inf
    train_losses = []
    val_losses = []
    best_model = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for idx, (batch_features, batch_Q_sqrt, batch_labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            weights = model(batch_features, batch_Q_sqrt)
            ret = torch.mul(weights, batch_labels)
            loss = -torch.sum(ret)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # if idx % 5 == 0:
            #     print('Current epoch: %d, Current batch: %d, Loss is %.3f' %(epoch+1,idx+1,loss.item()))

        epoch_loss /= len(train_dataloader)
        train_losses.append(epoch_loss)

        val_loss = test_model(val_dataloader, model)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss}, Validation Loss: {val_loss}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping:
            print("Early stopping")
            break
    return train_losses, val_losses, best_model

def test_model(dataloader, model):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for idx, (batch_features, batch_Q_sqrt, batch_labels) in enumerate(dataloader):
            weights = model(batch_features, batch_Q_sqrt)
            ret = torch.mul(weights, batch_labels)
            loss = -torch.sum(ret)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def rolling_train(data, window_size=1800, train_size=1500, val_size=300, roll_step=100, gap=21, epochs=50, early_stopping=10):
    levels = data.index.get_level_values(0).drop_duplicates()
    data = data.loc[levels]
    max_index = len(levels) - window_size - gap * 2 - roll_step
    i = 0
    predictions = pd.DataFrame()

    while i <= max_index:
        print(f"Rolling Window: {i}/{max_index}")
        
        window_dates = levels[i:i + window_size + gap * 2 + roll_step]
        train_dates = window_dates[:train_size]
        val_dates = window_dates[train_size + gap:train_size + gap + val_size]
        test_dates = window_dates[train_size + gap * 2 + val_size:train_size + gap * 2 + val_size + roll_step]

        train_data = data.loc[train_dates]
        val_data = data.loc[val_dates]
        test_data = data.loc[test_dates]

        train_dataset = FactorDataset(train_data)
        val_dataset = FactorDataset(val_data)
        test_dataset = FactorDataset(test_data)
        train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        # for batch_inputs, batch_cov_matrices, batch_labels in val_dataloader:
        #     # print(batch_inputs[0]) 
        #     # print(batch_cov_matrices[0])
        #     # print(batch_labels[0])
        #     print(batch_inputs.shape)       
        #     print(batch_cov_matrices.shape) 
        #     print(batch_labels.shape)
        #     break    

        model = FactorModel(input_dim=44, hidden_dim=10, output_dim=4, lower=0.05, upper=0.35)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_losses, val_losses, best_model = train_model(train_dataloader, val_dataloader, model, optimizer, epochs=epochs, early_stopping=early_stopping)
        torch.save(best_model, f'./test/{test_dates[-roll_step:][0].strftime("%Y%m%d")}-{test_dates[-1].strftime("%Y%m%d")},({round(train_losses[val_losses.index(min(val_losses))],2)},{round(min(val_losses),2)}).pth')

        plt.figure(figsize=(12, 8))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./test/learning_curve_{test_dates[-roll_step:][0].strftime("%Y%m%d")}-{test_dates[-1].strftime("%Y%m%d")}.png')
        plt.close()

        model.load_state_dict(best_model)
        model.eval()
        with torch.no_grad():
            for batch_features, batch_Q_sqrt, _ in test_dataloader:
                weights = model(batch_features, batch_Q_sqrt)
                weights = pd.DataFrame(weights.cpu().numpy(), index=test_data.index.get_level_values(0).drop_duplicates()[-roll_step:], columns=test_data.index.get_level_values(1).drop_duplicates())
                predictions = pd.concat([predictions, weights], axis=0)

        i += roll_step
        
    predictions.to_csv('./test/Weights.xlsx')

def main():
    set_seed(0)
    data = pd.read_parquet('test.parquet')
    rolling_train(data)

if __name__ == '__main__':
     main()