import torch
import torch.nn as nn
import time
import math
import numpy as np
from losses import QAGLoss

# Mock modules for the tasks
def run_regression_seed(seed, device):
    torch.manual_seed(seed)
    N, B = 1000, 64
    target = torch.randn(B, N, device=device)
    source = nn.Parameter(torch.zeros(B, N, device=device))
    opt = torch.optim.Adam([source], lr=0.1)
    loss_fn = QAGLoss().to(device)
    
    start = time.time()
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(source, target).mean()
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
    
    # Task 1
    t1_times, t1_losses = [], []
    for s in seeds:
        t, l = run_regression_seed(s, device)
        t1_times.append(t); t1_losses.append(l)
    print(f"Task 1 (Regression) | Time: {np.mean(t1_times):.3f}s ± {np.std(t1_times):.3f}s | Final Loss: {np.mean(t1_losses):.4f} ± {np.std(t1_losses):.4f}")

    # Task 4
    t4_times, t4_losses = [], []
    for s in seeds:
        t, l = run_sw_seed(s, device)
        t4_times.append(t); t4_losses.append(l)
    print(f"Task 4 (Sliced-W)   | Time: {np.mean(t4_times):.3f}s ± {np.std(t4_times):.3f}s | Final Loss: {np.mean(t4_losses):.4f} ± {np.std(t4_losses):.4f}")