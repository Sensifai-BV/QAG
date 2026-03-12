import torch
import torch.nn as nn
import ot  # POT library
from geomloss import SamplesLoss

class QAGLoss(nn.Module):
    """Exact 1D W2^2 Loss via Quantile Affine Geometry[cite: 9, 142, 163]."""
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_sorted, _ = torch.sort(x, dim=-1)
        y_sorted, _ = torch.sort(y, dim=-1)
        loss = torch.mean((x_sorted - y_sorted)**2, dim=-1)
        return loss.sum()

class SinkhornLoss(nn.Module):
    """GeomLoss Sinkhorn wrapper for 1D tensors."""
    def __init__(self, blur=0.05):
        super().__init__()
        self.loss_fn = SamplesLoss(
            loss="sinkhorn", 
            p=2, 
            blur=blur, 
            debias=False, 
            backend="tensorized" # (or "online", whichever you are testing!)
        )
    def forward(self, x, y):
        batch_size = x.shape[0]
        total_loss = 0.0
        
        # Explicitly loop over the batch to bypass GeomLoss's B>1 limitation
        for i in range(batch_size):
            # Slice to keep dimensions as (1, N, 1) so multiscale/online is happy
            x_i = x[i:i+1].unsqueeze(-1)
            y_i = y[i:i+1].unsqueeze(-1)
            
            total_loss += self.loss_fn(x_i, y_i)
            
        return total_loss


class POTLoss(nn.Module):
    """POT CPU baseline (Forward only)."""
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # Detach and move to CPU for POT 
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        batch_size = x_np.shape[0]
        n_samples = x_np.shape[1]
        
        # Uniform weights
        a, b = ot.unif(n_samples), ot.unif(n_samples)
        
        total_loss = 0.0
        for i in range(batch_size):
            # 1D Wasserstein squared
            loss_i = ot.wasserstein_1d(x_np[i], y_np[i], a, b, p=2)
            total_loss += loss_i
            
        return torch.tensor(total_loss, device=x.device)