import numpy as np
import torch

from worldmodel.model import MLP, Normalizer, WorldModel
from worldmodel.train import masked_mse, train_model


def test_normalizer_constant_column():
    a = np.c_[np.full(10, 5.0), np.arange(10.0)].astype(np.float32)
    n = Normalizer.fit(a)
    t = n.transform(a)
    assert np.isfinite(t).all()
    np.testing.assert_allclose(n.inverse(t), a, rtol=1e-6)


def test_masked_mse_ignores_masked():
    pred = torch.zeros(2, 3)
    tgt = torch.ones(2, 3)
    mask = torch.ones(2, 3)
    mask[0, 1] = 0
    a = masked_mse(pred, tgt, mask)
    tgt2 = tgt.clone()
    tgt2[0, 1] = 100.0
    assert torch.isclose(a, masked_mse(pred, tgt2, mask))
    assert torch.isclose(a, torch.tensor(1.0))


def _toy(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    Y = (X[:, :2] * 0.5).astype(np.float32)
    return X, Y, np.ones_like(Y)


def test_overfits_tiny():
    model, hist = train_model(_toy(64, 0), _toy(64, 1), epochs=400, lr=3e-3, batch=16, patience=400)
    X, Y, _ = _toy(64, 1)
    assert np.mean((model.predict_delta(X) - Y) ** 2) < 1e-2
    assert hist[-1]["val_loss"] < hist[0]["val_loss"]


def test_save_load_same_predictions(tmp_path):
    X, Y, _ = _toy(32, 0)
    m = WorldModel(MLP(4, 2), Normalizer.fit(X), Normalizer.fit(Y))
    m.save(tmp_path / "m.pt")
    m2 = WorldModel.load(tmp_path / "m.pt")
    np.testing.assert_allclose(m.predict_delta(X), m2.predict_delta(X))
    np.testing.assert_allclose(m.predict_next(X, X[:, :2]), X[:, :2] + m.predict_delta(X), rtol=1e-6)


def test_gru_overfits_tiny():
    """The target depends on the OLDEST window, so the model must use its memory."""
    def seq(n, seed):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 3, 4)).astype(np.float32)
        Y = (X[:, 0, :2] * 0.5).astype(np.float32)
        return X, Y, np.ones_like(Y)
    model, _ = train_model(seq(64, 0), seq(64, 1), epochs=600, lr=3e-3, batch=16, patience=600,
                           arch="gru", history=3)
    X, Y, _ = seq(64, 1)
    assert model.arch == "gru" and model.history == 3
    assert np.mean((model.predict_delta(X) - Y) ** 2) < 2e-2
