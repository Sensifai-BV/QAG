import torch
import numpy as np
import argparse
from losses import QAGLoss, SinkhornLoss, POTLoss # Unified imports

def time_and_memory(loss_module, x, y, bwd=False, warmups=50, runs=200):
    # Warmups 
    for _ in range(warmups):
        loss = loss_module(x, y)
        if bwd:
            loss.backward()
            x.grad.zero_()
            
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    times = []
    # Timed runs 
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        loss = loss_module(x, y)
        if bwd:
            loss.backward()
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        if bwd:
            x.grad.zero_()
            
    mem_gb = torch.cuda.max_memory_allocated() / (1024**3) # Peak memory in GB [cite: 9, 264, 284]
    return np.median(times), mem_gb

def main(args):
    device = torch.device("cuda")
    print(f"--- Section 7.2: Scaling & Memory Benchmark ---")
    
    qag, sinkhorn, pot = QAGLoss().to(device), SinkhornLoss().to(device), POTLoss().to(device)
    
    for n in args.n_sizes:
        print(f"\nEvaluating N={n}")
        x = torch.randn(args.batch_size, n, device=device, requires_grad=True)
        y = torch.randn(args.batch_size, n, device=device)
        
        # 1. POT (CPU, Fwd only) 
        try:
            t_pot, _ = time_and_memory(pot, x, y, bwd=False)
            print(f"POT (CPU Fwd): {t_pot:.2f} ms")
        except Exception:
            print("POT: Skipped/OOM")
            
        # 2. Sinkhorn [cite: 9, 285, 286]
        try:
            t_sink_f, mem_sink_f = time_and_memory(sinkhorn, x, y, bwd=False)
            t_sink_b, mem_sink_b = time_and_memory(sinkhorn, x, y, bwd=True)
            print(f"Sinkhorn (Fwd): {t_sink_f:.2f} ms | Mem: {mem_sink_f:.2f} GB")
            print(f"Sinkhorn (Bwd): {t_sink_b:.2f} ms | Mem: {mem_sink_b:.2f} GB")
        except RuntimeError:
            print("Sinkhorn: OOM")

        # 3. QAG [cite: 9, 286]
        try:
            t_qag_f, mem_qag_f = time_and_memory(qag, x, y, bwd=False)
            t_qag_b, mem_qag_b = time_and_memory(qag, x, y, bwd=True)
            print(f"QAG (Fwd): {t_qag_f:.2f} ms | Mem: {mem_qag_f:.2f} GB")
            print(f"QAG (Bwd): {t_qag_b:.2f} ms | Mem: {mem_qag_b:.2f} GB")
        except RuntimeError:
            print("QAG: OOM")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_sizes", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    main(parser.parse_args())