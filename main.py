import numpy as np
import torch.nn as nn
import torch
from Mmodel import Mmodel
from DTDataset import DTDataset
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve, r2_score, \
    mean_squared_error, mean_absolute_error

# %%
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 256
NUM_EPOCHS = 1000
LR = 0.0003
modeling = Mmodel
cuda_name = "cuda:0"

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data["smiles"] = data["smiles"].to(device)
        data["protein"] = data["protein"].to(device)
        data["label"] = data["label"].to(device).float()
        optimizer.zero_grad()
        output = model(data["smiles"],data["protein"],data["label"],TRAIN_BATCH_SIZE)
        preds = output
        loss = loss_fn(preds,data["label"])
        loss.backward()
        optimizer.step()
        return preds,data["label"]

def predicting(model, device, loader):
    model.eval()
    with torch.no_grad():
        for data in loader:
            data["smiles"] = data["smiles"].to(device)
            data["protein"] = data["protein"].to(device)
            data["label"] = data["label"].to(device).float()
            optimizer.zero_grad()
            output = model(data["smiles"],data["protein"],data["label"],TRAIN_BATCH_SIZE)
            preds = output.cpu()
    return preds,data["label"]

if __name__ == '__main__':
    n_train = len(DTDataset())
    split = n_train // 5
    np.random.seed(42)
    indices = np.random.choice(range(n_train), size=n_train, replace=False)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    train_loader = DataLoader(DTDataset(), sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
    test_loader = DataLoader(DTDataset(), sampler=test_sampler, batch_size=TEST_BATCH_SIZE)
    # %%
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(NUM_EPOCHS):
        GT, GP = train(model, device, train_loader, optimizer)
        G, P = predicting(model, device, test_loader)
        G, P = G.cpu(),P.cpu()
        r2 = r2_score(G, P)
        mse = mean_squared_error(G, P)
        mae = mean_absolute_error(G, P)
        print(epoch, ":")
        print("r2:", r2, ",mse:", mse, ",mae:", mae)




