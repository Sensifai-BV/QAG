import torch
import torch.nn as nn
import time
import math
import numpy as np
from losses import QAGLoss

class DistributionalRegressionMLP(nn.Module):
    def __init__(self, input_dim=5, output_quantiles=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_quantiles)
        )
    def forward(self, x):
        return self.net(x)

def run_regression_seed(seed, device):
    torch.manual_seed(seed)
    N_quantiles, B = 1000, 64
    
    # Genuine Regression Setup: Mapping X -> Y
    X_train = torch.randn(B, 5, device=device)
    # Target distribution is a Gaussian shifted by the sum of input features
    shifts = X_train.sum(dim=1, keepdim=True)
    Y_target = torch.randn(B, N_quantiles, device=device) + shifts
    
    model = DistributionalRegressionMLP(input_dim=5, output_quantiles=N_quantiles).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = QAGLoss().to(device)
    
    start = time.time()
    for _ in range(50):
        opt.zero_grad()
        Y_pred = model(X_train)
        loss = loss_fn(Y_pred, Y_target).mean()
        loss.backward()
        opt.step()
        
    return time.time() - start, loss.item()

def run_sw_seed(seed, device):
    torch.manual_seed(seed)
    N = 1000
    target = torch.randn(N, 2, device=device) * 2.0
    source = nn.Parameter(torch.randn(N, 2, device=device))
    opt = torch.optim.Adam([source], lr=0.5)
    loss_fn = QAGLoss().to(device)
    
    start = time.time()
    for _ in range(50):
        opt.zero_grad()
        theta = torch.rand(50, device=device) * 2 * math.pi
        dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        p_src, p_tgt = torch.matmul(dirs, source.T), torch.matmul(dirs, target.T)
        loss = loss_fn(p_src, p_tgt).mean()
        loss.backward()
        opt.step()
    return time.time() - start, loss.item()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- Change 5: Multi-Seed End-to-End Tasks (5 Seeds) ---")
    seeds = [42, 43, 44, 45, 46]
    
    t1_times, t1_losses = [], []
    for s in seeds:
        t, l = run_regression_seed(s, device)
        t1_times.append(t); t1_losses.append(l)
    print(f"Task 1 (Regression) | Time: {np.mean(t1_times):.3f}s ± {np.std(t1_times):.3f}s | Final Loss: {np.mean(t1_losses):.4f} ± {np.std(t1_losses):.4f}")

    t4_times, t4_losses = [], []
    for s in seeds:
        t, l = run_sw_seed(s, device)
        t4_times.append(t); t4_losses.append(l)
    print(f"Task 4 (Sliced-W)   | Time: {np.mean(t4_times):.3f}s ± {np.std(t4_times):.3f}s | Final Loss: {np.mean(t4_losses):.4f} ± {np.std(t4_losses):.4f}")