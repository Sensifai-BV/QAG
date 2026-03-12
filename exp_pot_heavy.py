import torch
import ot
import numpy as np
from losses import QAGLoss

def run_exactness_check(device):
    print("--- Fix 1: POT Exactness Validation (Including Heavy-Tailed) ---")
    print(f"{'Dist':<14} | {'N':<8} | {'POT Mean':<12} | {'QAG Mean':<12} | {'Abs Err':<12} | {'Rel Err':<12}")
    print("-" * 79)
    
    sizes = [10**3, 10**4, 10**5]
    distributions = ['Gaussian', 'Bimodal', 'Heavy-Tailed']
    loss_fn = QAGLoss().to(device)
    
    # We use Student-T with df=3 for heavy tails but finite variance
    student_t = torch.distributions.StudentT(df=3.0)
    
    for dist in distributions:
        for N in sizes:
            pot_vals, qag_vals = [], []
            
            for _ in range(10):
                if dist == 'Gaussian':
                    x = torch.randn(N, device=device)
                    y = torch.randn(N, device=device)
                elif dist == 'Bimodal':
                    x = torch.cat([torch.randn(N//2, device=device) - 3, torch.randn(N//2, device=device) + 3])
                    y = torch.cat([torch.randn(N//2, device=device) - 3, torch.randn(N//2, device=device) + 3])
                    x = x[torch.randperm(N)]
                    y = y[torch.randperm(N)]
                elif dist == 'Heavy-Tailed':
                    x = student_t.sample((N,)).to(device)
                    y = student_t.sample((N,)).to(device)
                
                # QAG W2^2 Cost
                qag_val = loss_fn(x.unsqueeze(0), y.unsqueeze(0)).item()
                
                # POT W2^2 Cost
                x_np = x.cpu().numpy()
                y_np = y.cpu().numpy()
                pot_val = ot.wasserstein_1d(x_np, y_np, p=2)
                
                pot_vals.append(pot_val)
                qag_vals.append(qag_val)
            
            pot_mean = np.mean(pot_vals)
            qag_mean = np.mean(qag_vals)
            abs_errs = np.abs(np.array(pot_vals) - np.array(qag_vals))
            rel_errs = abs_errs / (np.array(pot_vals) + 1e-9)
            
            print(f"{dist:<14} | {N:<8} | {pot_mean:<12.6f} | {qag_mean:<12.6f} | {np.mean(abs_errs):<12.2e} | {np.mean(rel_errs):<12.2e}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_exactness_check(device)
