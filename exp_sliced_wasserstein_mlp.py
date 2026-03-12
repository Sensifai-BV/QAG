import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from losses import QAGLoss, SinkhornLoss
import os
import logging 
import warnings

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"

# --- 1. Dataset Generation ---
def generate_data(n_samples, device):
    """Generates synthetic 4-class Gaussian input data in R^8[cite: 78, 141]."""
    X, Y = [], []
    samples_per_class = n_samples // 4
    # Defined class means [cite: 118]
    mu = [
        [2, 0, 0, 0, 0, 0, 0, 0],
        [-2, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
        [0, -2, 0, 0, 0, 0, 0, 0]
    ]
    for c in range(4):
        x_c = torch.randn(samples_per_class, 8, device=device) + torch.tensor(mu[c], device=device, dtype=torch.float32)
        y_c = torch.full((samples_per_class,), c, device=device, dtype=torch.long)
        X.append(x_c)
        Y.append(y_c)
    
    X = torch.cat(X)
    Y = torch.cat(Y)
    
    # Shuffle the dataset
    idx = torch.randperm(n_samples)
    return X[idx], Y[idx]

# --- 2. Target Latent Distributions ---
def sample_target_gmm(c, n_samples, device):
    """Samples from class-specific 2-mode Gaussian mixtures in R^16[cite: 141]."""
    m1 = torch.zeros(16, device=device)
    m2 = torch.zeros(16, device=device)
    
    # Distinct centers for each class [cite: 119, 120]
    if c == 0:
        m1[0], m2[1] = 3.0, 3.0
    elif c == 1:
        m1[0], m2[1] = -3.0, -3.0
    elif c == 2:
        m1[2], m2[3] = 3.0, 3.0
    elif c == 3:
        m1[2], m2[3] = -3.0, -3.0
        
    # 50/50 mixture [cite: 119]
    choices = torch.randint(0, 2, (n_samples,), device=device)
    centers = torch.where(choices.unsqueeze(1) == 0, m1, m2)
    
    # Variance of 0.25 (std of 0.5) [cite: 88, 119]
    noise = torch.randn(n_samples, 16, device=device) * 0.5
    return centers + noise

# --- 3. Model Definition ---
class MLP(nn.Module):
    """MLP: 8 -> 64 -> 64 -> 16 plus linear classifier head[cite: 90, 141]."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.clf_head = nn.Linear(16, 4)
        
    def forward(self, x):
        emb = self.net(x)
        logits = self.clf_head(emb)
        return emb, logits

# --- 4. Sliced Wasserstein Loss ---
def sliced_wasserstein_1d_projections(Z_c, T_c, K, base_loss_fn, device):
    """Computes 1D Wasserstein distances over K random projections[cite: 102]."""
    if len(Z_c) == 0:
        return 0.0
    # Sample K random unit vectors in R^16 [cite: 102]
    theta = torch.randn(16, K, device=device)
    theta = theta / torch.norm(theta, dim=0, keepdim=True)
    
    # Project both sets to 1D [cite: 102]
    a = torch.matmul(Z_c, theta)
    b = torch.matmul(T_c, theta)
    
    # Calculate base 1D loss across all K projections [cite: 102]
    a_T = a.t() 
    b_T = b.t() 
    
    return base_loss_fn(a_T, b_T).mean()

# --- 5. Main Training Loop ---
def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Method\tSlices (K)\tVal Accuracy (up)\tVal Class-SW (down)\tEpoch Time (s) (down)\tPeak GPU Mem (GB) (down)\tStatus")
    
    # Exact sweep and hyperparams requested [cite: 108, 111, 141]
    slice_counts = [16, 64, 256, 1024]
    methods = ['Sinkhorn', 'QAG-STE']
    n_epochs = 20
    batch_size = 256 
    lam = 1.0 
    seeds = [42, 43, 44]
    
    for method_name in methods:
        for K in slice_counts:
            val_accs, val_sws, epoch_times, mems = [], [], [], []
            status = "OK" 
            
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # 10k train, 2k val balanced [cite: 80]
                X_train, Y_train = generate_data(10000, device)
                X_val, Y_val = generate_data(2000, device)
                
                model = MLP().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                
                if method_name == 'Sinkhorn':
                    base_loss = SinkhornLoss(blur=0.05).to(device)
                else:
                    base_loss = QAGLoss().to(device)
                    
                torch.cuda.reset_peak_memory_stats(device)
                
                try:
                    time_per_epoch = []
                    for epoch in range(n_epochs):
                        model.train()
                        start_time = time.time()
                        
                        idx = torch.randperm(10000)
                        for b in range(0, 10000, batch_size):
                            batch_idx = idx[b:b+batch_size]
                            x_b, y_b = X_train[batch_idx], Y_train[batch_idx]
                            
                            optimizer.zero_grad()
                            emb, logits = model(x_b)
                            
                            # Classification loss on true class label [cite: 99]
                            loss_ce = F.cross_entropy(logits, y_b)
                            
                            # Class-conditional SW loss [cite: 99, 141]
                            loss_sw = 0.0
                            classes_in_batch = torch.unique(y_b)
                            for c in classes_in_batch:
                                mask = (y_b == c)
                                Z_c = emb[mask]
                                n_c = len(Z_c)
                                T_c = sample_target_gmm(c.item(), n_c, device)
                                loss_sw += sliced_wasserstein_1d_projections(Z_c, T_c, K, base_loss, device)
                            loss_sw = loss_sw / len(classes_in_batch)
                            
                            total_loss = loss_ce + lam * loss_sw
                            total_loss.backward()
                            optimizer.step()
                            
                        epoch_times.append(time.time() - start_time)
                    
                    # Validation Evaluation [cite: 112, 113]
                    model.eval()
                    with torch.no_grad():
                        emb_val, logits_val = model(X_val)
                        preds = logits_val.argmax(dim=1)
                        val_acc = (preds == Y_val).float().mean().item() * 100.0
                        
                        val_sw_total = 0.0
                        for c in range(4):
                            mask = (Y_val == c)
                            Z_c = emb_val[mask]
                            n_c = len(Z_c)
                            T_c = sample_target_gmm(c, n_c, device)
                            # Large fixed evaluation slice count of 4096 [cite: 113, 114]
                            val_sw_total += sliced_wasserstein_1d_projections(Z_c, T_c, 4096, QAGLoss().to(device), device).item()
                        val_sw_avg = val_sw_total / 4.0
                        
                    val_accs.append(val_acc)
                    val_sws.append(val_sw_avg)
                    mems.append(torch.cuda.max_memory_allocated(device) / (1024**3))
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        status = "OOM" # [cite: 115]
                        break
                    else:
                        raise e
                        
            if status == "OOM":
                print(f"{method_name}\t{K}\t-\t-\t-\t-\tOOM")
            else:
                avg_time = np.mean(epoch_times)
                if avg_time > 30.0:
                    status = "Slow" # [cite: 115]
                print(f"{method_name}\t{K}\t{np.mean(val_accs):.1f}\t{np.mean(val_sws):.3f}\t{avg_time:.1f}\t{np.max(mems):.1f}\t{status}")

if __name__ == '__main__':
    run_experiment()
