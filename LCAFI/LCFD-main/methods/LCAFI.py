import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class LabelChannelAttention(nn.Module):
    def __init__(self, num_features, num_labels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(num_labels, num_features // reduction)
        self.fc2 = nn.Linear(num_features // reduction, num_features)

    def forward(self, feature_emb, label_vec):
        attn = F.relu(self.fc1(label_vec))
        attn = torch.sigmoid(self.fc2(attn))
        return feature_emb * attn


class LCAFI(nn.Module):
    def __init__(
        self,
        num_features,
        num_labels,
        hidden_dim=128,
        attn_heads=4,
        lr=1e-3,
        weight_decay=1e-4,
        device='cuda:0',
        ema_momentum=0.9,
        topk=1,
        lambda_corr=0.02,
        prior_label_corr=None
    ):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.device = device
        self.ema_momentum = ema_momentum
        self.topk = topk
        self.lambda_corr = lambda_corr

        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.label_channel_attention = LabelChannelAttention(hidden_dim, num_labels)

        self.label_embedding = nn.Embedding(num_labels, hidden_dim)
        self.label_encoder = nn.Sequential(
            nn.Linear(num_labels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.label_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=attn_heads, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=attn_heads, batch_first=True
        )

        self.alignment_loss_weight = 0.1

        self.decoder_dom = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )
        self.decoder_tail = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )


        self.label_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )

        if prior_label_corr is not None:
            self.register_buffer('prior_R', torch.FloatTensor(prior_label_corr))
        else:
            self.register_buffer('prior_R', torch.eye(num_labels))

        self.register_buffer('ema_dom', torch.tensor(1.0))
        self.register_buffer('ema_tail', torch.tensor(1.0))

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.to(self.device)

    def forward(self, x, y):
        if x.ndimension() == 4:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        if y.ndimension() == 1:
            y = y.unsqueeze(1)
        feat_emb = self.feature_encoder(x)               # [B, D]
        label_vec = y                                   # [B, C]
        feat_emb = self.label_channel_attention(feat_emb, label_vec)
        feat_emb = feat_emb.unsqueeze(1)                 # [B, 1, D]
        label_emb = self.label_encoder(y).unsqueeze(1)   # [B, 1, D]
        cross_out, _ = self.cross_attn(
            query=label_emb, key=feat_emb, value=feat_emb
        )
        out = cross_out.squeeze(1)  # [B, D]
        alignment_loss = self.compute_alignment_loss(feat_emb.squeeze(1), label_emb.squeeze(1))
        logits_dom = self.decoder_dom(out)
        logits_tail = self.decoder_tail(out)
        dominant_mask, tail_mask = self.get_decouple_masks(y)
        true_label_logits = self.label_decoder(out)
        return logits_dom, logits_tail, dominant_mask, tail_mask, true_label_logits, alignment_loss

    def compute_alignment_loss(self, feat_emb, label_emb):
        cosine_similarity = F.cosine_similarity(feat_emb, label_emb, dim=1)
        alignment_loss = 1 - cosine_similarity.mean()
        return alignment_loss

    def get_decouple_masks(self, y):
 
        safe_y = y.clone()
        zero_mask = (y.sum(dim=1) == 0)  # [B]
        if zero_mask.any():
            safe_y[zero_mask, 0] = 1.0   

        _, top_idx = safe_y.topk(self.topk, dim=1)
        dominant_mask = torch.zeros_like(y)
        dominant_mask.scatter_(1, top_idx, 1.0)
        tail_mask = 1.0 - dominant_mask
        return dominant_mask, tail_mask

    def update_ema(self, ema_tensor, new_value):

        ema_tensor.mul_(self.ema_momentum).add_((1 - self.ema_momentum) * new_value)

    def compute_loss(self, logits_dom, logits_tail, dominant_mask, tail_mask, y, true_label_logits, alignment_loss, mask):
        preds_dom = F.softmax(logits_dom, dim=1).clamp(min=1e-8, max=1)
        preds_tail = F.softmax(logits_tail, dim=1).clamp(min=1e-8, max=1)
        y_norm = y / (y.sum(dim=1, keepdim=True) + 1e-8)
        effective_dom = dominant_mask * mask
        effective_tail = tail_mask * mask
        loss_dom = (-effective_dom * y_norm * torch.log(preds_dom)).sum() / effective_dom.sum().clamp(min=1)
        loss_tail = (-effective_tail * y_norm * torch.log(preds_tail)).sum() / effective_tail.sum().clamp(min=1)


        with torch.no_grad():
            self.update_ema(self.ema_dom, loss_dom.detach())
            self.update_ema(self.ema_tail, loss_tail.detach())
            ema_sum = self.ema_dom + self.ema_tail + 1e-8
            w_dom = self.ema_tail / ema_sum
            w_tail = self.ema_dom / ema_sum

        label_loss = F.cross_entropy(true_label_logits, y.argmax(dim=1))
        total_loss = (w_dom * loss_dom + w_tail * loss_tail
                      + 0.2 * label_loss
                      + 0.1 * alignment_loss)
        return total_loss

    def train_step(self, x, y, mask):
        self.train()
        outputs = self.forward(x, y)
        logits_dom, logits_tail, dominant_mask, tail_mask, true_label_logits, alignment_loss = outputs
        loss = self.compute_loss(
            logits_dom, logits_tail, dominant_mask, tail_mask, y, true_label_logits, alignment_loss, mask
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, save_path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def load(self, load_path, epoch=None):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from epoch {checkpoint['epoch']}")

    def get_result(self, test_loader):
        self.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                logits_dom, logits_tail, _, _, _, _ = self.forward(x_batch, y_batch)
                preds = (logits_dom + logits_tail) / 2
                preds = F.softmax(preds, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_preds, all_labels
