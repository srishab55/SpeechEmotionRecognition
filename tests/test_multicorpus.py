"""Fast checks that need no audio: model shape, speaker-disjoint folds, and the train/score path on synthetic frames."""

import numpy as np
import pandas as pd
import torch

from multicorpus import train
from unheard_lib import nets


def test_cnn_bilstm_shapes_and_padding_invariance():
    net = nets.build("cnn_bilstm_w2v", 16, 8, {"width": 8, "dropout": 0.0}).eval()
    x = torch.randn(2, 30, 16)
    x[1, 18:] = 0  # the cache pads with zeros
    lengths = torch.tensor([30, 18])
    logits, emb = net(x, lengths)
    assert logits.shape == (2, 8) and emb.shape == (2, 16)  # BiLSTM doubles the embedding width
    padded = torch.cat([x[1:, :18], torch.zeros(1, 12, 16)], 1)
    alone, _ = net(padded, torch.tensor([18]))
    assert torch.allclose(alone, logits[1:], atol=1e-5)


def _fake(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for corpus, n_spk in (("RAVDESS", 12), ("CREMA-D", 14)):
        for s in range(n_spk):
            for lab in range(4):
                for _ in range(2):
                    rows.append((corpus, f"{corpus}_{s}", lab))
    df = pd.DataFrame(rows, columns=["source", "speaker_id", "label_idx"])
    n = len(df)
    lengths = rng.integers(20, 40, n).astype(np.int32)
    x = rng.normal(size=(n, 40, 16)).astype(np.float16)
    for lab in range(4):  # learnable signal: each label lights up its own block of channels
        x[(df.label_idx == lab).to_numpy(), :, lab * 4 : lab * 4 + 4] += 3
    monkeypatch.setattr(train, "load_index", lambda: df)
    monkeypatch.setattr(train.frame_cache, "load", lambda: (x, lengths))
    monkeypatch.setattr(train, "OUT", tmp_path)
    monkeypatch.setattr(train, "HYPER", {"lr": 3e-3, "width": 8, "dropout": 0.1})
    monkeypatch.setattr(train, "SEEDS", (42,))
    return df


def test_folds_hold_out_whole_speakers(tmp_path, monkeypatch):
    df = _fake(tmp_path, monkeypatch)
    for tr, te in train.folds_for(df, 42):
        assert not set(df.speaker_id[tr]) & set(df.speaker_id[te])


def test_train_and_score_on_synthetic_frames(tmp_path, monkeypatch):
    df = _fake(tmp_path, monkeypatch)
    monkeypatch.setattr(train, "CORPORA", ["RAVDESS", "CREMA-D"])
    monkeypatch.setattr(train.frame_cache, "OUT", tmp_path / "x")
    train.run(max_epochs=12, patience=4, device="cpu")
    assert len(list(tmp_path.glob("pred_seed42_fold*.npz"))) == 5
    monkeypatch.setattr(train, "ROOT", train.ROOT)
    res = train.score(n_boot=20)
    assert res.loc[0, "set"] == "ALL four corpora" and res.loc[0, "macro_f1_mean_of_seeds"] > 0.4
