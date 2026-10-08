"""Neural network architectures for frame sequences.

Two families live here:

* **Sequence models** read hand-designed frames (log-mel or MFCC) and learn
  their own temporal filters: 1-D CNN, 2-D CNN, RNN, LSTM, GRU, BiLSTM, CNN-LSTM.
* **Attention models** read wav2vec2 frames and learn which frames matter:
  mean-pool MLP, additive attention, attentive statistics, a small
  Transformer, a uniform-weights control and a speaker-adversarial variant.

Every model takes ``x`` shaped ``(batch, frames, channels)`` and ``lengths``
(the real frame count of each clip) and returns ``(logits, embedding)``.
Clips are right-padded with zeros to a common length, so every step that mixes
frames (pooling, softmax, attention) uses the mask built from ``lengths``.
A short clip must give the same output whatever amount of padding follows it;
``tests/test_unheard_nets.py`` checks that for every model.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# --------------------------------------------------------------------------- masks and pooling


def length_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    """Boolean mask ``(batch, width)``: True on real frames, False on padding."""
    return torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]


def scaled_lengths(lengths: torch.Tensor, stride: int) -> torch.Tensor:
    """Real frame counts after a layer that shrinks time by ``stride`` (ceil, never below 1)."""
    return torch.clamp(torch.ceil(lengths.float() / stride).long(), min=1)


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over real frames only. ``h`` is ``(batch, frames, dim)``."""
    weights = mask.unsqueeze(-1).to(h.dtype)
    return (h * weights).sum(1) / weights.sum(1).clamp(min=1.0)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over frames in which padding receives exactly zero weight.

    Each row needs at least one real frame, which holds because every clip has
    a positive length.
    """
    return torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)


def weighted_stats(h: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Attention-weighted mean and standard deviation, concatenated."""
    w = weights.unsqueeze(-1)
    mean = (h * w).sum(1)
    var = (w * (h - mean.unsqueeze(1)) ** 2).sum(1).clamp(min=1e-6)
    return torch.cat([mean, var.sqrt()], dim=-1)


class ChannelNorm(nn.Module):
    """Layer normalisation over channels, applied to each frame separately.

    Batch normalisation would mix statistics across frames, and more than half
    of a padded batch is padding. Normalising each frame on its own cannot see
    padding, so the output for a clip does not depend on how much follows it.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: (batch, channels, frames)
        return self.norm(h.transpose(1, 2)).transpose(1, 2)


# --------------------------------------------------------------------------- sequence models


class CNN1D(nn.Module):
    """Convolutions along time over the feature channels, then masked mean pooling."""

    def __init__(self, in_dim: int, n_classes: int, width: int = 64, dropout: float = 0.3, **_) -> None:
        super().__init__()
        channels = [in_dim, width, width * 2, width * 2]
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=5, padding=2),
                ChannelNorm(cout),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for cin, cout in zip(channels[:-1], channels[1:], strict=True)
        )
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, lens = x.transpose(1, 2), lengths
        for block in self.blocks:
            # Zero the padding before pooling: a convolution puts its bias on padded frames, and a
            # pooling window that straddles the clip's end would otherwise read it.
            h = block(h) * length_mask(lens, h.shape[-1]).unsqueeze(1)
            h, lens = self.pool(h), scaled_lengths(lens, 2)
        embedding = masked_mean(h.transpose(1, 2), length_mask(lens, h.shape[-1]))
        return self.head(embedding), embedding


class CNN2D(nn.Module):
    """Convolutions over the time-frequency image, as in image classification."""

    def __init__(self, in_dim: int, n_classes: int, width: int = 32, dropout: float = 0.3, **_) -> None:
        super().__init__()
        channels = [1, width // 2, width, width * 2]
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(dropout / 2),
            )
            for cin, cout in zip(channels[:-1], channels[1:], strict=True)
        )
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, lens = x.transpose(1, 2).unsqueeze(1), lengths  # (batch, 1, channels, frames)
        for block in self.blocks:
            h = block(h) * length_mask(lens, h.shape[-1])[:, None, None, :]
            h, lens = self.pool(h), scaled_lengths(lens, 2)
        h = h.mean(2)  # average over the frequency axis: (batch, channels, frames)
        embedding = masked_mean(h.transpose(1, 2), length_mask(lens, h.shape[-1]))
        return self.head(embedding), embedding


class Recurrent(nn.Module):
    """RNN, LSTM, GRU or BiLSTM over packed sequences, masked mean over the outputs."""

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        kind: str = "lstm",
        bidirectional: bool = False,
        width: int = 64,
        dropout: float = 0.3,
        layers: int = 1,
        **_,
    ) -> None:
        super().__init__()
        cell = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cell(
            in_dim,
            width,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        out = width * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(out, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        embedding = masked_mean(out, length_mask(lengths, x.shape[1]))
        return self.head(self.drop(embedding)), embedding


class CNNLSTM(nn.Module):
    """Two convolution blocks that shorten time, then an LSTM over the result (BiLSTM if `bidirectional`)."""

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        width: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = False,
        project: bool = False,
        **_,
    ) -> None:
        super().__init__()
        # Wide inputs (768-d wav2vec2 frames) first go through the attention models' frame projection:
        # a convolution over 768 channels costs several times more than everything after it.
        self.proj = FrameProjection(in_dim, width, dropout) if project else None
        if project:
            in_dim = width
        self.convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(cin, width, kernel_size=5, padding=2),
                ChannelNorm(width),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for cin in (in_dim, width)
        )
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        self.lstm = nn.LSTM(width, width, batch_first=True, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width * (2 if bidirectional else 1), n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proj is not None:
            x = self.proj(x) * length_mask(lengths, x.shape[1]).unsqueeze(-1)
        h, lens = x.transpose(1, 2), lengths
        for block in self.convs:
            h = block(h) * length_mask(lens, h.shape[-1]).unsqueeze(1)
            h, lens = self.pool(h), scaled_lengths(lens, 2)
        packed = pack_padded_sequence(h.transpose(1, 2), lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=h.shape[-1])
        embedding = masked_mean(out, length_mask(lens, h.shape[-1]))
        return self.head(self.drop(embedding)), embedding


# --------------------------------------------------------------------------- attention models


class FrameProjection(nn.Module):
    """Frame-wise layer norm, linear projection and dropout: the shared front end of attention models."""

    def __init__(self, in_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#: Size of the clip embedding of the pooled models. Fixed so embeddings from different folds line up.
EMBED_DIM = 64


class Pooled(nn.Module):
    """A frame projection, a pooling rule and a small classifier.

    ``pooling`` selects how the frames become one vector:

    * ``mean``: masked mean (no attention, the pooled-embedding equivalent).
    * ``additive``: one learned score per frame, ``v . tanh(W h)``, softmaxed over real frames.
    * ``stats``: the same scores, but the output is the weighted mean and weighted standard deviation.
    * ``uniform``: the additive model with its scores replaced by equal weights. Same parameters,
      same code path, no ability to focus. This is the control that shows whether attention helps.

    After each forward pass ``self.weights`` holds the frame weights (or None for ``mean``).
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        pooling: str = "additive",
        width: int = 128,
        dropout: float = 0.3,
        **_,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "additive", "stats", "uniform"}:
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.front = FrameProjection(in_dim, width, dropout)
        self.score = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1))
        out = width * (2 if pooling == "stats" else 1)
        self.embed = nn.Sequential(nn.Linear(out, EMBED_DIM), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(EMBED_DIM, n_classes)
        self.weights: torch.Tensor | None = None

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            self.weights = None
            return masked_mean(h, mask)
        if self.pooling == "uniform":
            weights = mask.float() / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            weights = masked_softmax(self.score(h).squeeze(-1), mask)
        self.weights = weights.detach()
        if self.pooling == "stats":
            return weighted_stats(h, weights)
        return (h * weights.unsqueeze(-1)).sum(1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = length_mask(lengths, x.shape[1])
        return self.embed(self.pool(self.front(x), mask))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(x, lengths)
        return self.head(embedding), embedding


class TransformerPool(nn.Module):
    """A small Transformer encoder over frames, then masked mean pooling."""

    def __init__(
        self, in_dim: int, n_classes: int, width: int = 64, dropout: float = 0.2, layers: int = 2, **_
    ) -> None:
        super().__init__()
        width = max(16, (width // 4) * 4)  # divisible by the 4 attention heads
        self.front = FrameProjection(in_dim, width, dropout)
        layer = nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=width * 2, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(width, n_classes)
        self.width = width

    def _positions(self, frames: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(frames, device=device).float().unsqueeze(1)
        freq = torch.exp(
            torch.arange(0, self.width, 2, device=device).float() * (-math.log(10000.0) / self.width)
        )
        table = torch.zeros(frames, self.width, device=device)
        table[:, 0::2], table[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        return table

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = length_mask(lengths, x.shape[1])
        h = self.front(x) + self._positions(x.shape[1], x.device)
        h = self.encoder(h, src_key_padding_mask=~mask)
        embedding = masked_mean(h, mask)
        return self.head(embedding), embedding


# --------------------------------------------------------------------------- speaker-adversarial model


class _GradientReversal(torch.autograd.Function):
    """Identity on the forward pass, gradient multiplied by ``-strength`` on the way back."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, strength: float) -> torch.Tensor:  # noqa: ARG004
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.strength * grad, None


def reverse_gradient(x: torch.Tensor, strength: float) -> torch.Tensor:
    """Gradient reversal layer (Ganin and Lempitsky, 2015)."""
    return _GradientReversal.apply(x, strength)


class SpeakerAdversarial(Pooled):
    """Attentive-statistics model with a second head that tries to name the speaker.

    The speaker head sits behind a gradient reversal layer: it learns to
    recognise the actor, while the encoder underneath is pushed the opposite
    way, towards embeddings from which the actor cannot be told. Speaker
    labels come from the training actors of the current fold only.
    """

    def __init__(
        self, in_dim: int, n_classes: int, n_speakers: int, width: int = 128, dropout: float = 0.3, **_
    ) -> None:
        super().__init__(in_dim, n_classes, pooling="stats", width=width, dropout=dropout)
        self.n_speakers = n_speakers
        self.speaker_head = nn.Linear(EMBED_DIM, n_speakers)
        self.strength = 0.0  # set by the training loop; ramps up during training

    def speaker_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.speaker_head(reverse_gradient(embedding, self.strength))


class Standardised(nn.Module):
    """Wrap a model so its input is standardised with fixed channel statistics.

    Used by the normalisation ablation, where frames are not centred per speaker and are instead scaled with
    mean and standard deviation computed on the fitting clips only. Padded frames are set back to zero.
    """

    def __init__(self, core: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.core = core
        self.register_buffer("mean", mean.view(1, 1, -1))
        self.register_buffer("std", std.view(1, 1, -1))

    @property
    def weights(self) -> torch.Tensor | None:
        return getattr(self.core, "weights", None)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = (x - self.mean) / self.std * length_mask(lengths, x.shape[1]).unsqueeze(-1)
        return self.core(z, lengths)


# --------------------------------------------------------------------------- registries

SEQUENCE_MODELS: dict[str, tuple[type[nn.Module], dict]] = {
    "cnn1d": (CNN1D, {}),
    "cnn2d": (CNN2D, {}),
    "rnn": (Recurrent, {"kind": "rnn"}),
    "lstm": (Recurrent, {"kind": "lstm"}),
    "gru": (Recurrent, {"kind": "gru"}),
    "bilstm": (Recurrent, {"kind": "lstm", "bidirectional": True}),
    "cnn_lstm": (CNNLSTM, {}),
    "cnn_bilstm": (CNNLSTM, {"bidirectional": True}),
    "cnn_bilstm_w2v": (CNNLSTM, {"bidirectional": True, "project": True}),
}

ATTENTION_MODELS: dict[str, tuple[type[nn.Module], dict]] = {
    "mean_mlp": (Pooled, {"pooling": "mean"}),
    "uniform": (Pooled, {"pooling": "uniform"}),
    "additive": (Pooled, {"pooling": "additive"}),
    "attn_stats": (Pooled, {"pooling": "stats"}),
    "transformer": (TransformerPool, {}),
    "adversarial": (SpeakerAdversarial, {}),
}


def build(name: str, in_dim: int, n_classes: int, hyper: dict, n_speakers: int = 0) -> nn.Module:
    """Construct a registered model from its name and hyperparameters."""
    registry = SEQUENCE_MODELS | ATTENTION_MODELS
    cls, fixed = registry[name]
    kwargs = {**fixed, **{k: v for k, v in hyper.items() if k in {"width", "dropout", "layers"}}}
    if cls is SpeakerAdversarial:
        kwargs["n_speakers"] = n_speakers
    return cls(in_dim, n_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "ATTENTION_MODELS",
    "SEQUENCE_MODELS",
    "build",
    "count_parameters",
    "length_mask",
    "masked_mean",
    "masked_softmax",
    "Standardised",
    "reverse_gradient",
    "scaled_lengths",
]
