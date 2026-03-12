import torch
import time
from losses import QAGLoss, SinkhornLoss
import torch
import torch.nn as nn
import time
from losses import QAGLoss, SinkhornLoss

def run_histogram_matching(loss_fn, name, device, epochs=100):
    torch.manual_seed(42)
    N = 2000
    B = 1
    
    # Target: Bimodal (Sharp peaks at -3 and +3)
    target_left = torch.randn(B, N // 2, device=device) * 0.1 - 3.0
    target_right = torch.randn(B, N // 2, device=device) * 0.1 + 3.0
    target = torch.cat([target_left, target_right], dim=1)
    
    # Source: Uniform block
    particles = nn.Parameter(torch.rand(B, N, device=device) * 2 - 1)
    optimizer = torch.optim.Adam([particles], lr=0.1)
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(particles, target)
        loss.backward()
        optimizer.step()
        
    total_time = time.time() - start_time
    
    # Measure variance (True target variance is approx 9.0)
    final_variance = torch.var(particles).item()
    
    print(f"[{name}] Time: {total_time:.3f}s | Target Var: 9.00 | Result Var: {final_variance:.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- Experiment B: Histogram Matching (Peak Preservation) ---")
    run_histogram_matching(QAGLoss().to(device), "QAG Exact", device)
    # Using high blur to show the wash-out effect requested by reviewers
    run_histogram_matching(SinkhornLoss(blur=0.1).to(device), "Sinkhorn (blur=0.1)", device)