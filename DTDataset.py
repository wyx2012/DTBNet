import torch
import pandas as pd
class DTDataset():
    def __init__(self):
        super(DTDataset, self).__init__()
        csv_file_path = './data/davis_data.tsv'
        df = pd.read_csv(csv_file_path, sep='\t')
        smiles = df['drug']
        protein = df['protein']
        label = df['y']
        self.y = label
        dict_data = {'smiles': smiles,'protein': protein,'label': label}
        self.dict_data = dict_data

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.dict_data.items()}
        # item = {
        #     key: torch.tensor(value[idx]) if not isinstance(value[idx], torch.Tensor) else value[idx].clone().detach()
        #     for key, value in self.dict_data.items()}
        return item
    def __len__(self):
        return len(self.y)





