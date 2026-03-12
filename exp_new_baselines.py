import torch
import time
import torch.nn.functional as F
from losses import QAGLoss

def test_naive_baseline(device):
    print("--- Change 1: Naive PyTorch Baseline vs QAG Runtime ---")
    sizes = [10**3, 10**4, 10**5, 10**6]
    B = 64
    loss_fn = QAGLoss().to(device)
    
    for N in sizes:
        x = torch.randn(B, N, device=device, requires_grad=True)
        y = torch.randn(B, N, device=device)
        
        # Time Naive Exact PyTorch
        start = time.time()
        for _ in range(10):
            x_s, _ = torch.sort(x, dim=-1)
            y_s, _ = torch.sort(y, dim=-1)
            loss_naive = torch.mean((x_s - y_s)**2, dim=-1).sum()
            loss_naive.backward(retain_graph=True)
        torch.cuda.synchronize()
        naive_time = (time.time() - start) / 10 * 1000
        
        # Time QAG
        x.grad = None
        start = time.time()
        for _ in range(10):
            loss_qag = loss_fn(x, y).sum()
            loss_qag.backward(retain_graph=True)
        torch.cuda.synchronize()
        qag_time = (time.time() - start) / 10 * 1000
        
        print(f"N={N:<8} | Naive Exact: {naive_time:.2f} ms | QAG: {qag_time:.2f} ms")

def finite_difference_check(device):
    print("\n--- Change 2: Finite-Difference Gradient Check (N=32) ---")
    N = 32
    eps = 1e-4
    loss_fn = QAGLoss().to(device)
    
    def run_check(x, y, name):
        x.requires_grad_(True)
        loss = loss_fn(x.unsqueeze(0), y.unsqueeze(0)).sum()
        loss.backward()
        analytical_grad = x.grad.clone()
        
        fd_grad = torch.zeros_like(x)
        with torch.no_grad():
            for i in range(N):
                x_plus = x.clone()
                x_plus[i] += eps
                l_plus = loss_fn(x_plus.unsqueeze(0), y.unsqueeze(0)).sum()
                
                x_minus = x.clone()
                x_minus[i] -= eps
                l_minus = loss_fn(x_minus.unsqueeze(0), y.unsqueeze(0)).sum()
                
                fd_grad[i] = (l_plus - l_minus) / (2 * eps)
                
        cos_sim = F.cosine_similarity(analytical_grad.unsqueeze(0), fd_grad.unsqueeze(0)).item()
        rel_err = torch.norm(analytical_grad - fd_grad) / (torch.norm(fd_grad) + 1e-8)
        print(f"{name:<15} | Cosine Sim: {cos_sim:.6f} | Rel Norm Err: {rel_err:.6f}")

    # Continuous setting
    x_cont, y_cont = torch.randn(N, device=device), torch.randn(N, device=device)
    run_check(x_cont, y_cont, "Continuous")
    
    # Tie-heavy (quantized) setting
    x_tie, y_tie = torch.randint(0, 5, (N,), device=device).float(), torch.randint(0, 5, (N,), device=device).float()
    run_check(x_tie, y_tie, "Tie-heavy")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_naive_baseline(device)
    finite_difference_check(device)