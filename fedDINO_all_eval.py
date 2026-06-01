#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated Unsupervised SimCLR (Frozen DINOv3 Image Encoder)
Adapted from fedCLIP_all_eval.py — swaps the CLIP backbone for DINOv3 (via timm).
Default backbone is ViT-B/16 (vit_base_patch16_dinov3) to match CLIP-B/16 parameter
scale for a fair comparison. Preserves the paper's MLP projector design, NT-Xent
loss, and all evaluation modes. The readout averages all forward_features tokens
(CLS + 4 registers + 196 patches), matching the original fedDINOv3_all_eval.py
behavior so this script reproduces the paper's FedDINOv3 row at B/16 scale.
"""
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans
import os, json, random, copy, warnings
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split

from datasets_extra import HFImageDataset, HF_DATASETS
import timm
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from torchvision.datasets import ImageFolder

# Paper uses ImageNet normalization (Section 4.2). CLIP's own stats are kept here
# for reference / ablation but are not used by default.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ============================================================
# Utility helpers (unchanged from DINO version)
# ============================================================

def get_top_k_classes(cm, k=20):
    cm = cm.astype(float)
    correct = np.diag(cm)
    total = cm.sum(axis=1)
    acc = correct / np.maximum(total, 1e-8)
    top_k_idx = np.argsort(acc)[-k:][::-1]
    return top_k_idx, acc[top_k_idx]

def plot_confusion_matrix_zoom(cm, class_names, selected_idx, out_path,
                               title="Top-20 Confusion Matrix", figsize=12):
    sub_cm = cm[np.ix_(selected_idx, selected_idx)]
    n = sub_cm.shape[0]
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    im = ax.imshow(sub_cm, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([class_names[i] for i in selected_idx], rotation=90, fontsize=8)
    ax.set_yticklabels([class_names[i] for i in selected_idx], fontsize=8)
    ax.set_xlabel("Predicted Class", fontsize=14)
    ax.set_ylabel("True Class", fontsize=14)
    font_size = 12 if n <= 10 else 8 if n <= 20 else 6 if n <= 50 else 4
    for i in range(n):
        for j in range(n):
            val = sub_cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center", fontsize=font_size,
                    color="white" if val > sub_cm.max() * 0.5 else "black", alpha=0.9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved zoomed confusion matrix: {out_path}")


def get_tiny_imagenet_datasets(root="/home/ozgu/Desktop/Tiny_base_codes/tiny/", img_size=224):
    transform = EvalTransform(img_size)
    train_ds = ImageFolder(os.path.join(root, "train"), transform=transform)
    val_ds = ImageFolder(os.path.join(root, "val"), transform=transform)
    return train_ds, val_ds


def plot_confusion_matrix_pdf_large(y_true, y_pred, class_names, out_path,
                                    title="Confusion Matrix", figsize=30):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    im = ax.imshow(cm, aspect="auto", cmap="viridis")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{len(class_names)} predicted classes", fontsize=20)
    ax.set_ylabel(f"{len(class_names)} true classes", fontsize=20)
    n = cm.shape[0]
    font_size = 16 if n <= 10 else 10 if n <= 20 else 6 if n <= 50 else 4 if n <= 100 else 3
    for i in range(n):
        for j in range(n):
            value = cm[i, j]
            ax.text(j, i, str(value), ha="center", va="center", fontsize=font_size,
                    color="white" if value > cm.max() * 0.5 else "black", alpha=0.9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix: {out_path}")


def select_top_k_classes_for_tsne(labels, k=20):
    labels_np = np.asarray(labels)
    unique, counts = np.unique(labels_np, return_counts=True)
    top_k_classes = unique[np.argsort(counts)[-k:]]
    mask = np.isin(labels_np, top_k_classes)
    return mask, top_k_classes


import matplotlib.patches as mpatches

def plot_tsne_pdf_top20(embeddings, labels, out_path, title="Top-20 t-SNE", max_points=5000):
    emb = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
    lab = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    mask, top20_classes = select_top_k_classes_for_tsne(lab, k=20)
    emb = emb[mask]; lab = lab[mask]
    print(f"Using {emb.shape[0]} samples across top-20 classes: {top20_classes}")
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(8, 7))
    unique_labels = np.unique(lab)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=lab, s=6, cmap=cmap, alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    handles = [mpatches.Patch(color=cmap(i), label=str(cls)) for i, cls in enumerate(unique_labels)]
    ax.legend(handles=handles, title="Class IDs (Top-20)",
              bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved Top-20 t-SNE to: {out_path}")


def plot_tsne_pdf(embeddings, labels, out_path, title="t-SNE of Embeddings", max_points=None):
    emb = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
    lab = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    N = emb.shape[0]
    if max_points is not None and N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        emb = emb[idx]; lab = lab[idx]
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=lab, s=5, cmap="tab20")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved t-SNE plot to {out_path}")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_dataloader(ds, batch_size, shuffle, max_workers=4):
    nw = min(max_workers, os.cpu_count() or 2)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=nw, pin_memory=True, drop_last=False)

# ============================================================
# Shared DINOv3 backbone (singleton)
# ============================================================
class SharedDINOBackbone:
    _instance = None
    def __init__(self, cfg):
        if SharedDINOBackbone._instance: return
        model_name = cfg.dino_model
        print(f"Loading DINOv3 backbone ({model_name})...")
        bb = timm.create_model(model_name, pretrained=True)
        for p in bb.parameters(): p.requires_grad = False
        bb.eval()
        self.backbone = bb.to(cfg.device)
        self.feature_dim = getattr(bb, "embed_dim", 768)
        print(f"DINOv3 image feature dim: {self.feature_dim}")
        SharedDINOBackbone._instance = self

    @classmethod
    def get_instance(cls, cfg=None):
        if cls._instance is None: cls._instance = cls(cfg)
        return cls._instance

# ============================================================
# SimCLR model (frozen DINO backbone + same MLP projector as the paper)
# ============================================================
class UnsupervisedDINO(nn.Module):
    def __init__(self, shared_backbone, projection_dim=128):
        super().__init__()
        dim = shared_backbone.feature_dim
        self.shared_backbone = shared_backbone
        # Paper's projector: d_feat -> 512 (LN, GELU, Drop 0.2) -> 256 (LN, GELU, Drop 0.1) -> proj_dim
        self.projector = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, projection_dim),
        )

    def forward(self, x, return_features=False):
        # Match the original fedDINOv3_all_eval.py readout exactly:
        # forward_features returns [B, 201, D] (CLS + 4 register + 196 patches);
        # we mean-pool over all tokens. This is what produced the paper's
        # FedDINOv3 numbers, so it's the right baseline for fair comparison.
        f = self.shared_backbone.backbone.forward_features(x)
        if isinstance(f, dict):  # older timm DINOv2 / dict-style outputs
            f = f.get("x_norm_clstoken", f.get("x_norm_patchtokens", list(f.values())[0]))
        if f.dim() > 2:
            f = f.mean(1)
        z = self.projector(f)
        return (f, z) if return_features else z

# ============================================================
# Data transforms (using CLIP's normalization stats)
# ============================================================
def _get_norm(normalization):
    if normalization == "clip":
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD

class SimCLRViewTransform:
    def __init__(self, img=224, normalization="imagenet"):
        k = int(0.1 * img); k += 1 if k % 2 == 0 else 0
        mean, std = _get_norm(normalization)
        self.t = transforms.Compose([
            transforms.RandomResizedCrop(img, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    def __call__(self, x): return self.t(x), self.t(x)

class EvalTransform:
    def __init__(self, img=224, normalization="imagenet"):
        mean, std = _get_norm(normalization)
        self.t = transforms.Compose([
            transforms.Resize((img, img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    def __call__(self, x): return self.t(x)

# ============================================================
# NT-Xent loss
# ============================================================
class NTXentLoss(nn.Module):
    def __init__(self, t=0.5): super().__init__(); self.t = t
    def forward(self, z1, z2):
        N = z1.size(0); z = F.normalize(torch.cat([z1, z2], 0), dim=1)
        sim = torch.mm(z, z.t()) / self.t
        mask = torch.eye(2*N, device=z.device).bool(); sim.masked_fill_(mask, -float('inf'))
        pos = torch.cat([torch.arange(N, 2*N), torch.arange(N)], 0)
        pos_sim = sim[torch.arange(2*N), pos]
        logits = torch.cat([pos_sim.unsqueeze(1), sim], 1)
        labels = torch.zeros(2*N, dtype=torch.long, device=z.device)
        return F.cross_entropy(logits, labels)

# ============================================================
# FL client / server
# ============================================================
class FedSimCLRClient:
    def __init__(self, cid, model, loader, cfg):
        self.id = cid; self.model = model; self.loader = loader; self.cfg = cfg
        self.criterion = NTXentLoss(0.5)
        self.opt = optim.AdamW(model.projector.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    def train_local(self):
        self.model.train(); tot, ns = 0, 0
        n_batches = len(self.loader)
        for epoch in range(self.cfg.local_epochs):
            for bi, ((x1, x2), _) in enumerate(self.loader):
                x1, x2 = x1.to(self.cfg.device), x2.to(self.cfg.device)
                with torch.no_grad():
                    f1, _ = self.model(x1, return_features=True)
                    f2, _ = self.model(x2, return_features=True)
                z1, z2 = self.model.projector(f1), self.model.projector(f2)
                loss = self.criterion(z1, z2)
                self.opt.zero_grad()
                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.projector.parameters(), self.cfg.grad_clip)
                self.opt.step()
                tot += loss.item() * x1.size(0); ns += x1.size(0)
                if self.cfg.verbose and self.cfg.progress_every > 0 and ((bi + 1) % self.cfg.progress_every == 0):
                    print(f"    [client {self.id} epoch {epoch+1}/{self.cfg.local_epochs} batch {bi+1}/{n_batches}] running_loss={tot/max(ns,1):.4f}")
        return tot / ns

class FedServer:
    def __init__(self, gmodel): self.gmodel = gmodel
    def aggregate(self, models):
        sd = copy.deepcopy(self.gmodel.projector.state_dict())
        for k in sd.keys():
            sd[k] = torch.stack([m.projector.state_dict()[k].float() for m in models]).mean(0)
        self.gmodel.projector.load_state_dict(sd)

# ============================================================
# Embedding extraction + linear head + evaluations
# ============================================================
def extract_embeddings(model, ds, device, bs=128, verbose=False, progress_every=20, tag=""):
    loader = make_dataloader(ds, bs, False)
    n_batches = len(loader)
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            x = x.to(device)
            f, z = model(x, return_features=True)
            feats.append(z.cpu())  # projector output (matches DINO script)
            labels.append(y)
            if verbose and progress_every > 0 and ((bi + 1) % progress_every == 0):
                print(f"    [extract{(' ' + tag) if tag else ''} {bi+1}/{n_batches}]")
    return torch.cat(feats), torch.cat(labels)


class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes): super().__init__(); self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.fc(x)

def train_linear_head(trainf, trainy, valf, valy, testf, testy, in_dim, num_cls, device, cfg):
    model = LinearHead(in_dim, num_cls).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.linear_lr, weight_decay=cfg.linear_wd)
    lossf = nn.CrossEntropyLoss()

    train_loader = DataLoader(torch.utils.data.TensorDataset(trainf, trainy), cfg.linear_bs, True)
    val_loader = DataLoader(torch.utils.data.TensorDataset(valf, valy), cfg.linear_bs, False)
    test_loader = DataLoader(torch.utils.data.TensorDataset(testf, testy), cfg.linear_bs, False)

    best_val_acc = 0.0; patience_counter = 0; best_model_state = None

    for epoch in range(cfg.linear_epochs):
        model.train(); train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()

        model.eval(); val_corr, val_tot = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                val_corr += (pred == yb).sum().item(); val_tot += yb.size(0)
        val_acc = val_corr / val_tot

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= cfg.linear_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        if (epoch + 1) % 5 == 0:
            print(f"Linear Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc*100:.2f}%")

    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval(); corr, tot = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            corr += (pred == yb).sum().item(); tot += yb.size(0)
    return model, corr / tot, best_val_acc

def eval_linear_only(linear_model, testf, testy, device, save=False):
    linear_model.eval()
    loader = DataLoader(torch.utils.data.TensorDataset(testf, testy), 256, False)
    corr, tot = 0, 0; all_emb, all_lab = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = linear_model(xb); preds = logits.argmax(1)
            corr += (preds == yb).sum().item(); tot += yb.size(0)
            if save:
                all_emb.append(logits.cpu().numpy()); all_lab.append(yb.cpu().numpy())
    if save:
        os.makedirs("embeddings", exist_ok=True)
        np.save("embeddings/embeddings_evalonly.npy", np.concatenate(all_emb))
        np.save("embeddings/labels_evalonly.npy", np.concatenate(all_lab))
        print("Saved eval-only embeddings.")
    return corr / tot

def prototype_accuracy(embeddings, labels):
    emb_t = embeddings.float() if torch.is_tensor(embeddings) else torch.tensor(embeddings, dtype=torch.float32)
    labels_np = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)
    unique_classes = np.unique(labels_np)
    prototypes = []
    for c in unique_classes:
        mask = labels_np == c
        prototypes.append(emb_t[mask].mean(dim=0))
    prototypes = torch.stack(prototypes)
    sims = F.cosine_similarity(emb_t.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)
    preds = sims.argmax(dim=1).numpy()
    return accuracy_score(labels_np, preds)

# ============================================================
# Train-dataset dispatch (CIFAR-10/100 + any HF_DATASETS name)
# ============================================================
def _build_train_dataset(cfg, t_train, t_test):
    """Return (train_ds, test_ds, num_classes) for the requested cfg.dataset.

    - "cifar10" / "cifar100" use torchvision.
    - Any name registered in HF_DATASETS loads via HFImageDataset. The HF
      dataset must have a test_split for the held-out in-domain test set;
      if it doesn't, a 80/20 split of train is used.
    """
    if cfg.dataset == "cifar10":
        train_ds = torchvision.datasets.CIFAR10("./data", True, download=True, transform=t_train)
        test_ds = torchvision.datasets.CIFAR10("./data", False, download=True, transform=t_test)
        return train_ds, test_ds, 10
    if cfg.dataset == "cifar100":
        train_ds = torchvision.datasets.CIFAR100("./data", True, download=True, transform=t_train)
        test_ds = torchvision.datasets.CIFAR100("./data", False, download=True, transform=t_test)
        return train_ds, test_ds, 100
    if cfg.dataset in HF_DATASETS:
        entry = HF_DATASETS[cfg.dataset]
        train_ds = HFImageDataset.from_registry(cfg.dataset, split=entry.train_split, transform=t_train)
        if entry.test_split is None:
            raise ValueError(
                f"HF dataset '{cfg.dataset}' has no test_split registered. "
                f"Training requires a real test split for in-domain evaluation; "
                f"update HF_DATASETS in datasets_extra.py."
            )
        test_ds = HFImageDataset.from_registry(cfg.dataset, split=entry.test_split, transform=t_test)
        return train_ds, test_ds, len(train_ds.classes)
    raise ValueError(f"Unknown training dataset '{cfg.dataset}'. Known: cifar10, cifar100, {sorted(HF_DATASETS)}")


# ============================================================
# Main trainer
# ============================================================
class FedSimCLRTrainer:
    def __init__(self, cfg, balance=False):
        self.cfg = cfg; self.device = cfg.device
        os.makedirs("fl_results", exist_ok=True)
        t_train = SimCLRViewTransform(cfg.image_size, cfg.normalization)
        t_test = EvalTransform(cfg.image_size, cfg.normalization)

        train_ds, test_ds, self.num_classes = _build_train_dataset(cfg, t_train, t_test)
        self.test_ds = test_ds
        self.shared = SharedDINOBackbone.get_instance(cfg)
        self.global_model = UnsupervisedDINO(self.shared, projection_dim=cfg.projection_dim).to(self.device)
        self.server = FedServer(self.global_model)

        if cfg.dirichlet_alpha is not None:
            print(f"Using Dirichlet alpha={cfg.dirichlet_alpha} for non-IID partition...")
            if balance:
                print("Using balanced Dirichlet partition")
                client_idx = self._partition_dirichlet_balanced(train_ds, cfg.num_clients, cfg.dirichlet_alpha)
            else:
                print("Using imbalanced Dirichlet partition")
                client_idx = self._partition_dirichlet(train_ds, cfg.num_clients, cfg.dirichlet_alpha)
        else:
            idx = np.random.permutation(len(train_ds))
            splits = np.array_split(idx, cfg.num_clients)
            client_idx = [s.tolist() for s in splits]
            print("Using IID split across clients.")

        self.clients = []
        for cid, idxs in enumerate(client_idx):
            subset = Subset(train_ds, idxs)
            loader = make_dataloader(subset, cfg.batch_size, True)
            c_model = UnsupervisedDINO(self.shared, projection_dim=cfg.projection_dim).to(self.device)
            c_model.projector.load_state_dict(self.global_model.projector.state_dict())
            self.clients.append(FedSimCLRClient(cid, c_model, loader, cfg))

    def _partition_dirichlet(self, dataset, num_clients, alpha):
        labels = np.array(dataset.targets)
        num_cls = len(np.unique(labels))
        idx_by_cls = [np.where(labels == c)[0] for c in range(num_cls)]
        client_idx = [[] for _ in range(num_clients)]
        for c in range(num_cls):
            np.random.shuffle(idx_by_cls[c])
            props = np.random.dirichlet(alpha=[alpha] * num_clients)
            props = (np.cumsum(props) * len(idx_by_cls[c])).astype(int)[:-1]
            splits = np.split(idx_by_cls[c], props)
            for cid, s in enumerate(splits): client_idx[cid].extend(s)
        for cid in range(num_clients): np.random.shuffle(client_idx[cid])
        return client_idx

    def _partition_dirichlet_balanced(self, dataset, num_clients, alpha):
        labels = np.array(dataset.targets)
        num_classes = len(np.unique(labels))
        idx_by_cls = [np.where(labels == c)[0] for c in range(num_classes)]
        client_idx = [[] for _ in range(num_clients)]
        total_samples = len(labels)
        samples_per_client = total_samples // num_clients
        class_proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
        class_proportions = class_proportions / class_proportions.sum(axis=0, keepdims=True)
        for c in range(num_classes):
            np.random.shuffle(idx_by_cls[c])
            n_samples_cls = len(idx_by_cls[c])
            cls_sample_counts = (class_proportions[c] * n_samples_cls).astype(int)
            while cls_sample_counts.sum() < n_samples_cls:
                cls_sample_counts[np.argmin(cls_sample_counts)] += 1
            while cls_sample_counts.sum() > n_samples_cls:
                cls_sample_counts[np.argmax(cls_sample_counts)] -= 1
            start_idx = 0
            for cid in range(num_clients):
                end_idx = start_idx + cls_sample_counts[cid]
                client_idx[cid].extend(idx_by_cls[c][start_idx:end_idx])
                start_idx = end_idx
        for cid in range(num_clients):
            np.random.shuffle(client_idx[cid])
            if len(client_idx[cid]) > samples_per_client:
                client_idx[cid] = client_idx[cid][:samples_per_client]
            elif len(client_idx[cid]) < samples_per_client:
                diff = samples_per_client - len(client_idx[cid])
                extra = np.random.choice(total_samples, diff, replace=False)
                client_idx[cid].extend(extra.tolist())
            np.random.shuffle(client_idx[cid])
        return client_idx

    def train(self):
        print("Starting federated unsupervised SimCLR (CLIP backbone)...")
        results = {
            "rounds": [], "losses": [], "client_losses": [],
            "config": {k: str(v) for k, v in vars(self.cfg).items()}
        }
        best = float("inf")

        for r in range(self.cfg.rounds):
            print(f"\n--- Round {r+1}/{self.cfg.rounds} ---")
            sel = random.sample(self.clients, max(1, int(self.cfg.client_frac * len(self.clients))))
            cl_models, losses, client_ids = [], [], []
            for c in sel:
                l = c.train_local()
                losses.append(l); cl_models.append(c.model); client_ids.append(c.id)
                print(f"Client {c.id}: {l:.4f}")
            self.server.aggregate(cl_models)
            avg = np.mean(losses)
            results["rounds"].append(r + 1)
            results["losses"].append(float(avg))
            results["client_losses"].append({
                "round": r + 1, "client_ids": client_ids,
                "client_losses": [float(l) for l in losses]
            })
            best = min(best, avg)
            print(f"Round avg loss: {avg:.4f} (Best: {best:.4f})")
            if self.cfg.eval_every > 0 and (r + 1) % self.cfg.eval_every == 0:
                print("Running intermediate linear evaluation...")
                self._run_evaluations(results, r + 1)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{self.cfg.run_name}" if self.cfg.run_name else ""
        ckpt_path = (
            f"fl_results/checkpoints/"
            f"projector_{self.cfg.backbone_tag}_{self.cfg.dataset}{tag}_{ts}.pt"
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            "projector": self.global_model.projector.state_dict(),
            "config": {k: str(v) for k, v in vars(self.cfg).items()},
            "rounds_completed": self.cfg.rounds,
        }, ckpt_path)
        results["checkpoint_path"] = ckpt_path
        print(f"Projector checkpoint saved to {ckpt_path}")

        print("\nRunning final evaluations...")
        self._run_evaluations(results, self.cfg.rounds)
        out = f"fl_results/fed_unsup_simclr_{self.cfg.backbone_tag}_{self.cfg.dataset}{tag}_{ts}.json"
        with open(out, "w") as f: json.dump(results, f, indent=2)
        print(f"Results saved to {out}")

    def _build_eval_datasets(self, name):
        norm = self.cfg.normalization
        if name == "cifar10":
            train_full = torchvision.datasets.CIFAR10("./data", True, download=True, transform=EvalTransform(224, norm))
            test_eval = torchvision.datasets.CIFAR10("./data", False, download=True, transform=EvalTransform(224, norm))
            return train_full, test_eval, 10
        if name == "cifar100":
            train_full = torchvision.datasets.CIFAR100("./data", True, download=True, transform=EvalTransform(224, norm))
            test_eval = torchvision.datasets.CIFAR100("./data", False, download=True, transform=EvalTransform(224, norm))
            return train_full, test_eval, 100
        if name == "tiny":
            tiny_root = self.cfg.tiny_root
            train_full = ImageFolder(os.path.join(tiny_root, "train"), transform=EvalTransform(224, norm))
            test_eval = ImageFolder(os.path.join(tiny_root, "val"), transform=EvalTransform(224, norm))
            return train_full, test_eval, 200
        if name == "celeba":
            train_full = ImageFolder(self.cfg.celeba_root, transform=EvalTransform(224, norm))
            train_size = int(0.8 * len(train_full))
            test_size = len(train_full) - train_size
            train_full, test_eval = random_split(train_full, [train_size, test_size])
            return train_full, test_eval, 2
        # HuggingFace datasets registered in datasets_extra.HF_DATASETS.
        if name in HF_DATASETS:
            entry = HF_DATASETS[name]
            tfm = EvalTransform(224, norm)
            train_full = HFImageDataset.from_registry(name, split=entry.train_split, transform=tfm)
            n_classes = len(train_full.classes)
            if entry.test_split is not None:
                test_eval = HFImageDataset.from_registry(name, split=entry.test_split, transform=tfm)
            else:
                train_size = int(0.8 * len(train_full))
                test_size = len(train_full) - train_size
                train_full, test_eval = random_split(
                    train_full, [train_size, test_size],
                    generator=torch.Generator().manual_seed(42),
                )
            return train_full, test_eval, n_classes
        raise ValueError(f"Unknown eval_dataset: {name}")

    def _run_evaluations(self, results, round_num):
        # Cross-dataset eval: loop over self.cfg.eval_datasets (comma-separated CLI list).
        # Per the paper, after training on one dataset (e.g. CIFAR-10) we probe the
        # learned global model on multiple held-out datasets.
        for eval_name in self.cfg.eval_datasets:
            print(f"\n>>> Evaluating on '{eval_name}' <<<")
            self._run_one_eval(eval_name, results, round_num)

    def _run_one_eval(self, eval_name, results, round_num):
        train_full, test_eval, eval_num_classes = self._build_eval_datasets(eval_name)

        num_train = int(0.8 * len(train_full))
        num_val = len(train_full) - num_train
        train_subset, val_subset = random_split(
            train_full, [num_train, num_val],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"[{eval_name}] Linear split: {len(train_subset)} train, {len(val_subset)} val, {len(test_eval)} test")

        ee_kw = dict(verbose=self.cfg.verbose, progress_every=self.cfg.progress_every)
        trainf, trainy = extract_embeddings(self.global_model, train_subset, self.device, self.cfg.batch_size, tag=f"{eval_name} train", **ee_kw)
        valf, valy = extract_embeddings(self.global_model, val_subset, self.device, self.cfg.batch_size, tag=f"{eval_name} val", **ee_kw)
        testf, testy = extract_embeddings(self.global_model, test_eval, self.device, self.cfg.batch_size, tag=f"{eval_name} test", **ee_kw)

        lin_model, lin_acc, val_acc = train_linear_head(
            trainf, trainy, valf, valy, testf, testy,
            in_dim=self.cfg.projection_dim,
            num_cls=eval_num_classes,
            device=self.device,
            cfg=self.cfg,
        )
        print(f"[{eval_name}] Linear accuracy: {lin_acc*100:.2f}% | Val accuracy: {val_acc*100:.2f}%")

        lin_rand = LinearHead(self.cfg.projection_dim, eval_num_classes).to(self.device)
        acc_eval = eval_linear_only(lin_rand, testf, testy, self.device, save=self.cfg.save_test_embeddings)
        print(f"[{eval_name}] Eval-only accuracy: {acc_eval*100:.2f}%")

        self._run_unsupervised_evaluations(trainf, trainy, testf, testy, results, round_num, eval_num_classes, eval_name)

        if "eval_results" not in results:
            results["eval_results"] = []
        results["eval_results"].append({
            "round": round_num,
            "eval_dataset": eval_name,
            "linear_accuracy": float(lin_acc),
            "linear_val_accuracy": float(val_acc),
            "eval_only_accuracy": float(acc_eval),
            "timestamp": datetime.now().isoformat()
        })

    def _run_unsupervised_evaluations(self, trainf, trainy, testf, testy, results, round_num, eval_num_classes, eval_name="-"):
        trainf_n = F.normalize(trainf, dim=1)
        testf_n = F.normalize(testf, dim=1)

        print("Running unsupervised k-NN evaluation...")
        knn = KNeighborsClassifier(n_neighbors=20, metric="cosine")
        knn.fit(trainf_n, trainy)
        knn_pred = knn.predict(testf_n)
        knn_acc = accuracy_score(testy, knn_pred)
        print(f"k-NN Top-1 accuracy: {knn_acc*100:.2f}%")

        print("Running unsupervised clustering evaluation...")
        kmeans = KMeans(n_clusters=eval_num_classes, n_init=10, random_state=42)
        cluster_ids = kmeans.fit_predict(testf_n)
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import confusion_matrix as _cm
        cm = _cm(testy, cluster_ids)
        row_ind, col_ind = linear_sum_assignment(-cm)
        acc_cluster = cm[row_ind, col_ind].sum() / len(testy)
        print(f"K-Means clustering accuracy (Hungarian matched): {acc_cluster*100:.2f}%")

        try:
            sil = silhouette_score(testf_n, cluster_ids)
            print(f"Silhouette score: {sil:.3f}")
        except Exception as e:
            sil = None
            print(f"Silhouette score failed: {e}")

        proto_acc = prototype_accuracy(testf, testy)
        print(f"Prototype accuracy: {proto_acc*100:.2f}%")

        if "unsup_results" not in results:
            results["unsup_results"] = []
        results["unsup_results"].append({
            "round": round_num,
            "eval_dataset": eval_name,
            "knn_accuracy": float(knn_acc),
            "cluster_accuracy": float(acc_cluster),
            "prototype_accuracy": float(proto_acc),
            "silhouette_score": float(sil) if sil is not None else None
        })

# ============================================================
# Entry point
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10",
                   help="Training dataset. Accepts cifar10, cifar100, or any name "
                        "in HF_DATASETS (e.g. syke_ifcb, plankto_share).")

    # DINOv3 backbone
    p.add_argument("--dino_model", type=str, default="vit_base_patch16_dinov3",
                   help="timm model name. Defaults to ViT-B/16 to match CLIP-B/16 scale. "
                        "Use vit_small_patch16_dinov3 for the original paper's ViT-S/16.")

    p.add_argument("--projection_dim", type=int, default=128, help="SimCLR projector output dim")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_clients", type=int, default=2000)
    p.add_argument("--client_frac", type=float, default=0.05)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm (0 to disable)")

    # Linear evaluation
    p.add_argument("--linear_lr", type=float, default=1e-3)
    p.add_argument("--linear_wd", type=float, default=1e-4)
    p.add_argument("--linear_bs", type=int, default=256)
    p.add_argument("--linear_epochs", type=int, default=50)
    p.add_argument("--linear_patience", type=int, default=10)

    # Data partitioning
    p.add_argument("--dirichlet_alpha", type=float, default=0.3)
    p.add_argument("--balanced_partition", action="store_true", default=False,
                   help="Paper uses unbalanced Dirichlet (default). Pass this flag to force balanced.")
    p.add_argument("--normalization", type=str, default="imagenet", choices=["imagenet", "clip"],
                   help="Image normalization stats. Paper uses ImageNet; CLIP option for ablation.")
    p.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility / multi-seed runs.")
    p.add_argument("--backbone_tag", type=str, default="dino_b16",
                   help="Short tag used in result filenames and the aggregator regex "
                        "(e.g. dino_b16 for ViT-B/16, dino_s16 for ViT-S/16).")
    p.add_argument("--run_name", type=str, default="",
                   help="Tag included in output JSON filename (e.g. '200C_1pct_seed42').")
    p.add_argument("--verbose", action="store_true",
                   help="Print periodic progress (batch loss in train_local, embedding extraction batches).")
    p.add_argument("--progress_every", type=int, default=20,
                   help="Print a progress line every N batches when --verbose is set.")

    # Evaluation
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--save_test_embeddings", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--eval_datasets", type=str, default="cifar10",
                   help="Comma-separated list of eval datasets (cross-dataset eval). "
                        "Choices: cifar10, cifar100, tiny, celeba. "
                        "Example: --eval_datasets cifar10,cifar100,tiny")
    p.add_argument("--tiny_root", type=str, default="./tiny",
                   help="Path to Tiny-ImageNet root (needs train/ and val/ subfolders)")
    p.add_argument("--celeba_root", type=str, default="./celeba",
                   help="Path to CelebA ImageFolder root (binary attribute)")

    args = p.parse_args()
    args.eval_datasets = [s.strip() for s in args.eval_datasets.split(",") if s.strip()]
    builtin_evals = {"cifar10", "cifar100", "tiny", "celeba"}
    valid_evals = builtin_evals | set(HF_DATASETS.keys())
    for name in args.eval_datasets:
        if name not in valid_evals:
            raise ValueError(f"Invalid eval dataset '{name}'. Choose from {sorted(valid_evals)}.")
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {args.device}")
    print(f"Configuration: {vars(args)}")
    set_seed(args.seed)

    trainer = FedSimCLRTrainer(args, balance=args.balanced_partition)
    trainer.train()

if __name__ == "__main__":
    main()
