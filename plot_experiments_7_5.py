import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from losses import QAGLoss, SinkhornLoss

def run_quick_histogram(device):
    """Quickly runs Exp B to get the arrays for plotting."""
    torch.manual_seed(42)
    N, B = 2000, 1
    
    # Target: Bimodal (Sharp peaks)
    target_left = torch.randn(B, N // 2, device=device) * 0.1 - 3.0
    target_right = torch.randn(B, N // 2, device=device) * 0.1 + 3.0
    target = torch.cat([target_left, target_right], dim=1)
    
    # Train QAG
    p_qag = nn.Parameter(torch.rand(B, N, device=device) * 2 - 1)
    opt_qag = torch.optim.Adam([p_qag], lr=0.1)
    loss_qag = QAGLoss().to(device)
    for _ in range(100):
        opt_qag.zero_grad()
        loss_qag(p_qag, target).backward()
        opt_qag.step()
        
    # Train Sinkhorn
    p_sink = nn.Parameter(torch.rand(B, N, device=device) * 2 - 1)
    opt_sink = torch.optim.Adam([p_sink], lr=0.1)
    loss_sink = SinkhornLoss(blur=0.1).to(device)
    for _ in range(100):
        opt_sink.zero_grad()
        loss_sink(p_sink, target).backward()
        opt_sink.step()
        
    return target[0].detach().cpu().numpy(), p_qag[0].detach().cpu().numpy(), p_sink[0].detach().cpu().numpy()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Generating plots...")
    target, qag_res, sink_res = run_quick_histogram(device)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1
    methods_A = ['QAG (Exact)', 'Sinkhorn (blur=0.05)']
    times_A = [0.169, 30.515]
    barsA = axes[0].bar(methods_A, times_A, color=['#2ca02c', '#d62728'])

    axes[0].set_ylabel('Training Time (seconds)')
    axes[0].set_title('Exp A: Quantile Calibration\n(Lower is Better)')
    axes[0].set_yscale('log')

    for i, (bar, v) in enumerate(zip(barsA, times_A)):
        x = bar.get_x() + bar.get_width()/2
        if i == 1:  # red bar -> text inside
            axes[0].text(x, v/1.6, f"{v}s",
                         ha='center', va='top',
                         color='white', fontweight='bold')
        else:       # green bar -> text above
            axes[0].text(x, v*1.3, f"{v}s",
                         ha='center', va='bottom',
                         fontweight='bold')

    # Panel 2
    axes[1].hist(target, bins=60, density=True, alpha=0.3, color='gray', label='Target Distribution')
    axes[1].hist(qag_res, bins=60, density=True, histtype='step', linewidth=2, color='#2ca02c', label='QAG (Exact Matching)')
    axes[1].hist(sink_res, bins=60, density=True, histtype='step', linewidth=2, color='#d62728', linestyle='--', label='Sinkhorn (Washed Out)')
    axes[1].set_title('Exp B: Histogram Mode Preservation\nTarget Var: 9.00 | QAG: 9.01 | Sink: 9.18')
    axes[1].legend(loc='upper center')

    # Panel 3
    methods_C = ['QAG Inner Loop', 'Sinkhorn Inner Loop']
    times_C = [0.265, 23.341]
    barsC = axes[2].bar(methods_C, times_C, color=['#2ca02c', '#d62728'])

    axes[2].set_ylabel('Training Time (seconds)')
    axes[2].set_title('Exp C: Sliced-Wasserstein Flow\n(Lower is Better)')
    axes[2].set_yscale('log')

    for i, (bar, v) in enumerate(zip(barsC, times_C)):
        x = bar.get_x() + bar.get_width()/2
        if i == 1:  # red bar -> inside
            axes[2].text(x, v/1.6, f"{v}s",
                         ha='center', va='top',
                         color='white', fontweight='bold')
        else:       # green bar -> outside
            axes[2].text(x, v*1.3, f"{v}s",
                         ha='center', va='bottom',
                         fontweight='bold')

    plt.tight_layout()
    plt.savefig('sinkhorn_vs_qag.png', dpi=300)
    print("Plot successfully saved as 'sinkhorn_vs_qag.png'!")

if __name__ == "__main__":
    main()
