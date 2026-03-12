import torch
import torch.nn as nn
import time
from losses import QAGLoss, SinkhornLoss
import torch
import torch.nn as nn
import time
from losses import QAGLoss, SinkhornLoss

def run_calibration(loss_fn, name, device, epochs=50):
    torch.manual_seed(42)
    N = 1000
    B = 64
    
    # Ground truth (target) and raw predictions (source)
    target = torch.randn(B, N, device=device) * 2.0 + 5.0  # Mean 5, Std 2
    source = torch.randn(B, N, device=device)              # Mean 0, Std 1
    
    # Learnable calibration map: y = w * x + b
    w = nn.Parameter(torch.tensor(1.0, device=device))
    b = nn.Parameter(torch.tensor(0.0, device=device))
    optimizer = torch.optim.Adam([w, b], lr=0.1)
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        calibrated_source = w * source + b
        loss = loss_fn(calibrated_source, target)
        loss.backward()
        optimizer.step()
        
    total_time = time.time() - start_time
    
    # Calculate exact continuous ECE proxy (Absolute error of mean and std)
    final_mean = b.item()
    final_std = w.item()
    ece_proxy = abs(final_mean - 5.0) + abs(final_std - 2.0)
    
    print(f"[{name}] Time: {total_time:.3f}s | Final Loss: {loss.item() / B:.4f} | ECE Proxy: {ece_proxy:.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- Experiment A: Quantile Calibration ---")
    run_calibration(QAGLoss().to(device), "QAG Exact", device)
    # Using blur=0.05 as the balanced Sinkhorn baseline
    run_calibration(SinkhornLoss(blur=0.05).to(device), "Sinkhorn (blur=0.05)", device)