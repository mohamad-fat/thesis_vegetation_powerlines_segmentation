"""
Step 3: Point Transformer V2 Model
Simplified implementation based on the paper's best model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def knn(x, k):
    """Find k nearest neighbors"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


class TransformerBlock(nn.Module):
    """Point Transformer Block"""
    
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-head attention components
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        
        # Position encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, features, positions, k=16):
        """
        features: (B, N, d_model)
        positions: (B, N, 3)
        """
        B, N, C = features.shape
        
        # Get k-nearest neighbors
        idx = knn(positions.transpose(1, 2), k)  # (B, N, k)
        
        # Gather neighbor features
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)
        neighbor_features = torch.gather(
            features.unsqueeze(2).expand(-1, -1, k, -1),
            1,
            idx_expanded
        )  # (B, N, k, C)
        
        # Position encoding
        positions_expanded = positions.unsqueeze(2).expand(-1, -1, k, -1)
        neighbor_positions = torch.gather(
            positions.unsqueeze(2).expand(-1, -1, k, -1),
            1,
            idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )
        relative_pos = positions_expanded - neighbor_positions
        pos_encoding = self.pos_mlp(relative_pos)  # (B, N, k, C)
        
        # Attention
        q = self.q(features).unsqueeze(2)  # (B, N, 1, C)
        k = self.k(neighbor_features)  # (B, N, k, C)
        v = self.v(neighbor_features)  # (B, N, k, C)
        
        # Add position encoding
        k = k + pos_encoding
        v = v + pos_encoding
        
        # Attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)  # (B, N, 1, k)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v).squeeze(2)  # (B, N, C)
        
        # Residual + norm
        features = self.norm1(features + out)
        
        # FFN
        out = self.ffn(features)
        features = self.norm2(features + out)
        
        return features


class PointTransformerV2(nn.Module):
    """Simplified Point Transformer V2 for semantic segmentation"""
    
    def __init__(self, num_classes=6, d_model=256, num_layers=4):
        super().__init__()
        self.num_classes = num_classes
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model) for _ in range(num_layers)
        ])
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, points):
        """
        points: (B, N, 3)
        returns: (B, N, num_classes)
        """
        # Embed points
        features = self.input_embed(points)  # (B, N, d_model)
        
        # Apply transformer blocks
        for block in self.blocks:
            features = block(features, points)
        
        # Segmentation
        logits = self.seg_head(features)  # (B, N, num_classes)
        
        return logits


class PointTransformerLightning(L.LightningModule):
    """Lightning module for Point Transformer training"""
    
    def __init__(
        self,
        num_classes=6,
        d_model=256,
        num_layers=4,
        lr=1e-3,
        weight_decay=1e-4,
        class_weights=None,
        max_epochs=200
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = PointTransformerV2(
            num_classes=num_classes,
            d_model=d_model,
            num_layers=num_layers
        )
        
        # Loss with class weights (critical from paper!)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            # Default weights from paper's approach
            # Increase weights for underrepresented classes (tower, power line)
            self.class_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 5.0, 5.0])
        
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-1)
        
        # Metrics
        self.train_acc = []
        self.val_acc = []
    
    def forward(self, points):
        return self.model(points)
    
    def calculate_iou(self, pred, target, num_classes):
        """Calculate per-class IoU"""
        ious = []
        pred = pred.view(-1)
        target = target.view(-1)
        
        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()
            
            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append((intersection / union).item())
        
        return ious
    
    def training_step(self, batch, batch_idx):
        points = batch['points']  # (B, N, 3)
        labels = batch['labels']  # (B, N)
        
        # Forward
        logits = self(points)  # (B, N, num_classes)
        
        # Loss
        loss = self.criterion(
            logits.reshape(-1, self.hparams.num_classes),
            labels.reshape(-1)
        )
        
        # Accuracy
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean()
        
        # Log
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        points = batch['points']
        labels = batch['labels']
        
        # Forward
        logits = self(points)
        
        # Loss
        loss = self.criterion(
            logits.reshape(-1, self.hparams.num_classes),
            labels.reshape(-1)
        )
        
        # Metrics
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean()
        
        # IoU
        ious = self.calculate_iou(pred, labels, self.hparams.num_classes)
        mean_iou = np.nanmean([iou for iou in ious if not np.isnan(iou)])
        
        # Log
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_miou', mean_iou, prog_bar=True)
        
        return {'val_loss': loss, 'val_miou': mean_iou}
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


if __name__ == "__main__":
    # Test model
    model = PointTransformerLightning(num_classes=6)
    
    # Dummy input
    points = torch.randn(2, 4096, 3)
    
    # Forward pass
    logits = model(points)
    print("Output shape:", logits.shape)  # Should be (2, 4096, 6)