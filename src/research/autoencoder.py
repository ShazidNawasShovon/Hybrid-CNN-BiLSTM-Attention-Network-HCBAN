import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_sizes=(128, 64)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.enc = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, n_features),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)


def fit_autoencoder_recon_error(
    X_train,
    X_eval,
    device,
    epochs=10,
    batch_size=512,
    lr=1e-3,
    max_train_samples=50000,
):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_eval = np.asarray(X_eval, dtype=np.float32)

    if X_train.shape[0] > max_train_samples:
        idx = np.random.RandomState(42).choice(X_train.shape[0], size=max_train_samples, replace=False)
        X_train = X_train[idx]

    model = MLPAutoencoder(n_features=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='mean')

    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(int(epochs)):
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_eval, dtype=torch.float32).to(device)
        out = model(x)
        err = torch.mean((out - x) ** 2, dim=1).detach().cpu().numpy()
    return err
