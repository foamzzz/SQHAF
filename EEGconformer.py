"""EEG-Conformer for standard supervised LOSO evaluation.

The model keeps the original EEG-Conformer CNN + Transformer + classifier
workflow.  It does not perform target calibration or target fine-tuning.
The notebook supplies the training subjects and the held-out test subject.
"""

from __future__ import annotations

from contextlib import nullcontext
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PatchEmbedding(nn.Module):
    """Original EEG-Conformer convolutional patch embedding."""

    def __init__(self, n_channels: int, emb_size: int = 40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x.flatten(start_dim=2).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    """Original EEG-Conformer multi-head self-attention block."""

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        b, n, _ = x.shape
        head_dim = self.emb_size // self.num_heads
        queries = self.queries(x).view(b, n, self.num_heads, head_dim).transpose(1, 2)
        keys = self.keys(x).view(b, n, self.num_heads, head_dim).transpose(1, 2)
        values = self.values(x).view(b, n, self.num_heads, head_dim).transpose(1, 2)

        energy = torch.matmul(queries, keys.transpose(-2, -1))
        if mask is not None:
            energy = energy.masked_fill(~mask, torch.finfo(energy.dtype).min)
        att = F.softmax(energy / (self.emb_size ** 0.5), dim=-1)
        att = self.att_drop(att)
        out = torch.matmul(att, values)
        out = out.transpose(1, 2).contiguous().view(b, n, self.emb_size)
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_heads=10,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        super().__init__()
        self.attention = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )
        )
        self.feed_forward = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, forward_expansion, forward_drop_p),
                nn.Dropout(forward_drop_p),
            )
        )

    def forward(self, x):
        x = self.attention(x)
        return self.feed_forward(x)


class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size, num_heads=10):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(emb_size, num_heads=num_heads) for _ in range(depth)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ClassificationHead(nn.Module):
    """Original flatten -> 256 -> 32 classifier, generalized to n classes."""

    def __init__(self, feature_dim, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)


class Conformer(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes, emb_size=40, depth=6, n_heads=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(n_channels, emb_size)
        self.transformer = TransformerEncoder(depth, emb_size, n_heads)
        n_time_tokens = (n_samples - 25 + 1 - 75) // 15 + 1
        if n_time_tokens < 1:
            raise ValueError("n_samples must be at least 99 for the original EEG-Conformer pooling.")
        self.classifier = ClassificationHead(emb_size * n_time_tokens, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        return self.classifier(x)


class EEGConformer:
    """Original-style supervised EEG-Conformer with a Notebook-friendly API.

    This class performs ordinary supervised training on the supplied training
    data and predicts the supplied test data. It has no source/target domain
    adaptation and no target calibration or target fine-tuning.
    """

    def __init__(
        self,
        srate,
        freqs,
        *,
        device=None,
        emb_size=40,
        depth=6,
        n_heads=10,
        batch_size=72,
        n_epochs=2000,
        lr=2e-4,
        beta1=0.5,
        beta2=0.999,
        seed=42,
        use_amp=True,
    ):
        self.srate = float(srate)
        self.freqs = np.asarray(freqs, dtype=float)
        self.emb_size = int(emb_size)
        self.depth = int(depth)
        self.n_heads = int(n_heads)
        self.batch_size = int(batch_size)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.seed = int(seed)
        self.use_amp = bool(use_amp)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        self.model = None
        self.train_mean = None
        self.train_std = None
        self.n_channels = None
        self.n_samples = None

    def _autocast(self):
        if self.device.type == "cuda" and self.use_amp:
            return torch.cuda.amp.autocast()
        return nullcontext()

    def _loader(self, X, y, shuffle):
        return DataLoader(
            TensorDataset(
                torch.from_numpy(X).unsqueeze(1),
                torch.from_numpy(np.asarray(y, dtype=np.int64)),
            ),
            batch_size=min(self.batch_size, len(y)),
            shuffle=shuffle,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=int)
        if X_train.ndim != 3 or y_train.ndim != 1 or len(X_train) != len(y_train):
            raise ValueError("X_train must be (trials, channels, samples) and y_train must match it.")
        if np.any(y_train < 0) or np.any(y_train >= len(self.freqs)):
            raise ValueError("Training labels are outside the frequency index range.")

        _set_seed(self.seed)
        self.n_channels = int(X_train.shape[1])
        self.n_samples = int(X_train.shape[2])
        self.train_mean = float(X_train.mean())
        self.train_std = float(max(X_train.std(), 1e-6))
        X_train = (X_train - self.train_mean) / self.train_std

        self.model = Conformer(
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            n_classes=len(self.freqs),
            emb_size=self.emb_size,
            depth=self.depth,
            n_heads=self.n_heads,
        ).to(self.device)

        loader = self._loader(X_train, y_train, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(
            enabled=self.device.type == "cuda" and self.use_amp
        )

        self.model.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    loss = criterion(self.model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        return self

    def predict(self, X_test, batch_size=512):
        if self.model is None:
            raise RuntimeError("Call fit before predict.")
        X_test = np.asarray(X_test, dtype=np.float32)
        if X_test.ndim != 3 or X_test.shape[1:] != (self.n_channels, self.n_samples):
            raise ValueError(
                f"X_test must have shape (trials, {self.n_channels}, {self.n_samples})."
            )
        X_test = (X_test - self.train_mean) / self.train_std
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test).unsqueeze(1),
                torch.zeros(len(X_test), dtype=torch.long),
            ),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        pred = []
        self.model.eval()
        with torch.inference_mode():
            for xb, _ in loader:
                xb = xb.to(self.device, non_blocking=True)
                with self._autocast():
                    logits = self.model(xb)
                pred.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(pred).astype(int)
