import torch
import torch.nn as nn
import ot  # POT library
from geomloss import SamplesLoss


class QAG_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target):
        pred_sorted, pred_indices = torch.sort(pred, dim=-1)
        target_sorted, _ = torch.sort(target, dim=-1)
        
        ctx.save_for_backward(pred_indices, pred_sorted, target_sorted)
        
        # Returns a per-sample distance tensor of shape (B,)
        loss = torch.mean((pred_sorted - target_sorted) ** 2, dim=-1)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        pred_indices, pred_sorted, target_sorted = ctx.saved_tensors
        
        # Reshape upstream gradient for broadcasting: (B,) -> (B, 1)
        grad_output = grad_output.unsqueeze(-1)
        
        grad_sorted = 2.0 * (pred_sorted - target_sorted) / pred_sorted.shape[-1]
        
        grad_pred = torch.zeros_like(pred_sorted)
        grad_pred.scatter_(-1, pred_indices, grad_sorted)
        
        return grad_pred * grad_output, None

class QAGLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return QAG_STE.apply(pred, target)

class SinkhornLoss(nn.Module):
    def __init__(self, blur=0.05):
        super().__init__()
        self.loss_fn = SamplesLoss(
            loss="sinkhorn", 
            p=2, 
            blur=blur, 
            debias=False, 
            backend="tensorized" 
        )
    def forward(self, x, y):
        batch_size = x.shape[0]
        losses = []
        for i in range(batch_size):
            x_i = x[i:i+1].unsqueeze(-1)
            y_i = y[i:i+1].unsqueeze(-1)
            # Collect scalar loss per batch element
            losses.append(self.loss_fn(x_i, y_i).squeeze())
            
        return torch.stack(losses) # Returns shape (B,)


class POTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        batch_size = x_np.shape[0]
        n_samples = x_np.shape[1]
        
        a, b = ot.unif(n_samples), ot.unif(n_samples)
        
        losses = []
        for i in range(batch_size):
            loss_i = ot.wasserstein_1d(x_np[i], y_np[i], a, b, p=2)
            losses.append(loss_i)
            
        return torch.tensor(losses, device=x.device, dtype=torch.float32) # Returns shape (B,)