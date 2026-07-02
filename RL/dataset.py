from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, csv_file, columns_to_remove, fc_cols=None, scaler=None):
        self.data = pd.read_csv(csv_file)
        self.data_raw = self.data.copy()
        self.orig_columns = self.data.columns
        self.columns_to_remove = columns_to_remove
        self.fc_cols = fc_cols
        
        # Keep raw metrics for FC channel simulation and evaluation
        if "BLER_UL" in self.data.columns:
            self.raw_bler = self.data["BLER_UL"].values
        else:
            self.raw_bler = np.zeros(len(self.data))
            
        if "bw_avg" in self.data.columns:
            self.raw_bw = self.data["bw_avg"].values
        else:
            self.raw_bw = np.zeros(len(self.data))
            
        if "smoothed_rtt" in self.data.columns:
            self.raw_rtt = self.data["smoothed_rtt"].values
        else:
            self.raw_rtt = np.zeros(len(self.data))
            
        if scaler is not None:
            self.scaler = scaler
            self.fit_scaler = False
        else:
            self.scaler = StandardScaler()
            self.fit_scaler = True
            
        self._preprocess_data()

    def _preprocess_data(self):
        self.timestamps = self.data["timestamp"].values

        if self.fit_scaler:
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.data = self.scaler.transform(self.data)
            
        cw_index = self.orig_columns.get_loc("cw_avg")
        self.cw_mean = self.scaler.mean_[cw_index]
        self.cw_var = self.scaler.var_[cw_index]

        self.data = pd.DataFrame(self.data, columns=self.orig_columns)
        self.data = self.data.drop(columns=self.columns_to_remove)

        self.y = self.data["cw_avg"].values.reshape(-1, 1)
        self.original_y = torch.tensor((self.y * np.sqrt(self.cw_var)) + self.cw_mean, dtype=torch.float32)
        
        if self.fc_cols is not None:
            self.data = self.data[self.fc_cols]
        else:
            self.data = self.data.drop(columns=["cw_avg"])
            
        self.data = torch.tensor(self.data.values, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.original_y[idx]
    
    def get_cw_mean_and_var(self):
        return self.cw_mean, self.cw_var
    
    def get_original_y(self):
        return self.original_y
    
    def get_original_from_output(self, output):
        return torch.tensor((output * np.sqrt(self.cw_var)) + self.cw_mean, dtype=torch.float32)
    
    def shuffle_data(self):
        permutation_indices = torch.randperm(len(self.data))

        self.data = self.data[permutation_indices]
        self.original_y = self.original_y[permutation_indices]
        self.timestamps = self.timestamps[permutation_indices]
        
        # Shuffle raw tracking arrays identically
        if hasattr(self, 'raw_bler'):
            self.raw_bler = self.raw_bler[permutation_indices.numpy()]
        if hasattr(self, 'raw_bw'):
            self.raw_bw = self.raw_bw[permutation_indices.numpy()]
        if hasattr(self, 'raw_rtt'):
            self.raw_rtt = self.raw_rtt[permutation_indices.numpy()]

        self.y = self.y[permutation_indices]

    def get_timestamps(self, idx):
        return self.timestamps[idx]