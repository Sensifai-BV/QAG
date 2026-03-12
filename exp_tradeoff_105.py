import torch
import time
from losses import QAGLoss, SinkhornLoss

def run_tradeoff(device):
    print("--- Change 4: Accuracy-Runtime Tradeoff (N=100,000) ---")
    N = 10**5
    B = 64
    x = torch.randn(B, N, device=device)
    y = torch.randn(B, N, device=device)
    
    # Exact Baseline
    start = time.time()
    qag_loss = QAGLoss().to(device)(x, y).mean().item()
    torch.cuda.synchronize()
    qag_time = (time.time() - start) * 1000
    print(f"QAG Exact (Reference) | Time: {qag_time:.2f} ms | W2^2: {qag_loss:.4f} | Rel Err: 0.00%")
    
    # Sinkhorn sweep
    blurs = [0.01, 0.02, 0.05, 0.1]
    for b in blurs:
        loss_fn = SinkhornLoss(blur=b).to(device)
        start = time.time()
        s_loss = loss_fn(x, y).mean().item()
        torch.cuda.synchronize()
        s_time = (time.time() - start) * 1000
        err = abs(s_loss - qag_loss) / qag_loss * 100
        print(f"Sinkhorn (blur={b:<4}) | Time: {s_time:.2f} ms | W2^2: {s_loss:.4f} | Rel Err: {err:.2f}%")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_tradeoff(device)