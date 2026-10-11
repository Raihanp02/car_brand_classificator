import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

# --------------------------------------------------------------------------
# 4. Model: DINOv2 (with registers) backbone + projection head
#
# Embedding methods (must stay in sync with backend/app/embedder.py METHODS):
#     cls_proj256       MLP-projected CLS token, dim 256 (default, trainable head)
#     cls               raw CLS token, dim 768 (backbone fine-tune only)
#     patch_mean        mean of patch tokens, dim 768 (backbone fine-tune only)
#     cls_patch_concat  concat(CLS, patch_mean), dim 1536 (backbone fine-tune only)
# --------------------------------------------------------------------------
METHODS = {"cls_proj256": 256, "cls": 768, "patch_mean": 768, "cls_patch_concat": 1536}

class DinoV2Embedder(nn.Module):
    def __init__(self, backbone_name="facebook/dinov2-with-registers-base",
                 embed_dim=256, freeze_backbone_layers=0,
                 method="cls_proj256"):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {sorted(METHODS)}")
        if embed_dim != METHODS[method]:
            raise ValueError(f"embed_dim={embed_dim} incompatible with method={method!r} (dim {METHODS[method]})")
        self.method = method
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
 
        # optionally freeze the first N transformer blocks (cheaper fine-tune,
        # good starting point given 8GB VRAM)
        if freeze_backbone_layers > 0:
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = False
            for layer in self.backbone.encoder.layer[:freeze_backbone_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
 
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        ) if method == "cls_proj256" else None

    def head_parameters(self):
        """Parameters of the trainable head (empty for raw methods)."""
        return self.proj.parameters() if self.proj is not None else []

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0]        # CLS embedding
        if self.method == "cls":
            emb = cls_token
        elif self.method == "cls_proj256":
            emb = self.proj(cls_token)
        else:
            # NOTE: mean over h[:, 1:] includes register tokens — this must
            # match backend/app/embedder.py exactly so train/serve agree.
            patch_mean = out.last_hidden_state[:, 1:].mean(dim=1)
            emb = torch.cat([cls_token, patch_mean], dim=-1) if self.method == "cls_patch_concat" else patch_mean
        emb = F.normalize(emb, dim=-1)
        return emb
 
 
# --------------------------------------------------------------------------
# 5. Loss: SubCenter-ArcFace (closed-set, K sub-centers per class absorbs
#    intra-class variance e.g. camera angle) -- swap for TripletLoss below
#    if you want open-set gallery matching instead.
# --------------------------------------------------------------------------
class SubCenterArcFace(nn.Module):
    def __init__(self, embed_dim, num_classes, sub_centers=3, margin=0.35, scale=64):
        super().__init__()
        self.sub_centers = sub_centers
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes * sub_centers, embed_dim))
        nn.init.xavier_uniform_(self.weight)
 
    def forward(self, embeddings, labels):
        W = F.normalize(self.weight, dim=-1)
        cosine_all = embeddings @ W.t()                              # (B, C*K)
        cosine_all = cosine_all.view(-1, cosine_all.size(1) // self.sub_centers, self.sub_centers)
        cosine, _ = cosine_all.max(dim=-1)                            # (B, C) best sub-center
 
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logit = torch.cos(theta + self.margin)
 
        one_hot = F.one_hot(labels, cosine.size(1)).float()
        logits = cosine * (1 - one_hot) + target_logit * one_hot
        logits *= self.scale
        return F.cross_entropy(logits, labels)
 
 
class TripletLoss(nn.Module):
    """Batch-hard triplet loss -- use this instead if you go open-set."""
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
 
    def forward(self, embeddings, labels):
        dist = torch.cdist(embeddings, embeddings, p=2)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff = ~same
 
        hardest_pos = (dist * same.float()).max(dim=1).values
        masked_dist = dist.clone()
        masked_dist[same] = float("inf")
        hardest_neg = masked_dist.min(dim=1).values
 
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()

def triplet_loss_with_stats(embeddings, labels, margin):
    dist = torch.cdist(embeddings, embeddings, p=2)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
 
    hardest_pos = (dist * same.float()).max(dim=1).values
    masked_dist = dist.clone()
    masked_dist[same] = float("inf")
    hardest_neg = masked_dist.min(dim=1).values
 
    raw = hardest_pos - hardest_neg + margin
    loss = F.relu(raw)
    frac_active = (raw > 0).float().mean().item()   # fraction of triplets still contributing loss
    return loss.mean(), frac_active
