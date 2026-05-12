"""Smoke tests — verify modules import and core algorithms run on synthetic data.

These tests DO NOT need any real audio data, real attack models, or GPU.
They run a tiny end-to-end pipeline with a DummyTTS generator + numpy noise
and verify shapes, math, and CLI plumbing.

Run with:
    python -m voice_defense.tests.test_smoke
"""

from __future__ import annotations
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _wav_pair(dir_: Path, name: str, n_files: int, sr: int = 16000) -> None:
    """Write n_files of 2-second sine-noise wavs into dir_/<name>."""
    import soundfile as sf
    d = dir_ / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
        wav = 0.1 * np.sin(2 * np.pi * (200 + i * 5) * t)
        wav += 0.01 * np.random.randn(len(wav)).astype(np.float32)
        sf.write(d / f"{name}_{i:03d}.wav", wav, sr)


def test_metrics() -> None:
    from voice_defense.evaluation.metrics import compute_eer, compute_min_tdcf
    rng = np.random.default_rng(0)
    bonafide = rng.normal(loc=1.0, scale=0.5, size=200)
    spoof = rng.normal(loc=-1.0, scale=0.5, size=200)
    scores = np.concatenate([bonafide, spoof])
    labels = np.array([1] * 200 + [0] * 200)
    eer, thr = compute_eer(scores, labels)
    tdcf, _ = compute_min_tdcf(scores, labels)
    assert 0.0 <= eer <= 0.2, f"EER on well-separated synth should be small, got {eer}"
    assert 0.0 <= tdcf <= 2.0
    print(f"[metrics] EER={eer*100:.2f}% min-tDCF={tdcf:.4f}  OK")


def test_attack_zoo_dummy() -> None:
    from voice_defense.attack_zoo.base import DummyTTS
    from voice_defense.attack_zoo.orchestrator import AttackOrchestrator

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        orch = AttackOrchestrator([DummyTTS()], post_processors=[])
        prompts = [{"text": "hello"} for _ in range(3)]
        written = orch.generate_batch(prompts=prompts, out_dir=td, n_total=3)
        assert len(written) == 3
        meta_files = list(td.glob("*.json"))
        assert len(meta_files) == 3
        meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
        assert meta["algorithm"] == "dummy_tts"
        print(f"[attack_zoo] dummy + orchestrator wrote {len(written)} samples  OK")


def test_protocol_loader_and_dataset() -> None:
    import yaml
    from voice_defense.data_pipeline.dataset import load_protocol, ProtocolDataset

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _wav_pair(td, "bona", 4)
        _wav_pair(td, "fake_a", 3)
        _wav_pair(td, "fake_b", 3)

        proto = {
            "name": "tmp",
            "description": "smoke",
            "bonafide": [{"path": "bona", "glob": "*.wav"}],
            "spoof": [
                {"path": "fake_a", "attack_tag": "alpha"},
                {"path": "fake_b", "attack_tag": "beta"},
            ],
        }
        proto_path = td / "tmp.yaml"
        proto_path.write_text(yaml.safe_dump(proto), encoding="utf-8")

        spec = load_protocol(proto_path, base_dir=td)
        assert len(spec) == 10, f"expected 10 entries, got {len(spec)}"
        ds = ProtocolDataset(spec, segment_seconds=1.0, train=False)
        wav, label, tag = ds[0]
        assert wav.shape[0] == 16000
        assert label in (0, 1)
        # at least one each
        labels = [ds[i][1] for i in range(len(ds))]
        assert 1 in labels and 0 in labels
        tags = {ds[i][2] for i in range(len(ds))}
        assert {"bonafide", "alpha", "beta"}.issubset(tags)
        print(f"[data_pipeline] protocol + dataset OK ({len(ds)} samples, tags={tags})  OK")


def test_rawboost_runs() -> None:
    from voice_defense.data_pipeline.augment import RawBoost
    rb = RawBoost()
    x = (np.random.randn(16000).astype(np.float32) * 0.1)
    for _ in range(5):
        y = rb(x, 16000)
        assert y.shape == x.shape
        assert np.isfinite(y).all()
    print(f"[augment] RawBoost runs (3 algos)  OK")


def test_loss_math() -> None:
    import torch
    from voice_defense.defense.loss import OCSoftmaxLoss
    loss_fn = OCSoftmaxLoss(feat_dim=8)
    emb = torch.randn(16, 8)
    labels = torch.randint(0, 2, (16,))
    loss, scores = loss_fn(emb, labels)
    assert loss.dim() == 0
    assert scores.shape == (16,)
    assert torch.isfinite(loss)
    # cosine sim ranges [-1, 1]
    assert (scores >= -1.0 - 1e-5).all() and (scores <= 1.0 + 1e-5).all()
    print(f"[loss] OC-Softmax loss={loss.item():.4f}, scores in [-1,1]  OK")


def test_aasist_backend_shape() -> None:
    import torch
    from voice_defense.defense.backend import AASIST
    backend = AASIST(in_dim=64, gat_dim=32, n_subgraph_nodes=8, embed_dim=16)
    x = torch.randn(2, 50, 64)  # (B, T, D)
    z = backend(x)
    assert z.shape == (2, 16), z.shape
    # gradient flows
    z.sum().backward()
    assert any(p.grad is not None for p in backend.parameters())
    print(f"[backend] AASIST out={tuple(z.shape)} with grad  OK")


def test_full_pipeline_no_ssl() -> None:
    """Compose backend + OC-Softmax on random "frontend" features, verify a
    training step reduces loss. SSL frontend skipped to keep test offline."""
    import torch
    from voice_defense.defense.backend import AASIST
    from voice_defense.defense.loss import OCSoftmaxLoss

    torch.manual_seed(0)
    backend = AASIST(in_dim=32, gat_dim=16, n_subgraph_nodes=8, embed_dim=12)
    loss_fn = OCSoftmaxLoss(feat_dim=12)
    opt = torch.optim.Adam(
        list(backend.parameters()) + list(loss_fn.parameters()), lr=1e-2
    )

    # Synthetic: two clusters in feature space; bonafide and spoof.
    B = 32
    bona_feat = torch.randn(B // 2, 20, 32) + 1.5
    spoof_feat = torch.randn(B // 2, 20, 32) - 1.5
    x = torch.cat([bona_feat, spoof_feat], dim=0)
    y = torch.cat([torch.ones(B // 2, dtype=torch.long),
                   torch.zeros(B // 2, dtype=torch.long)])

    initial_loss = None
    for step in range(50):
        emb = backend(x)
        loss, _ = loss_fn(emb, y)
        if step == 0:
            initial_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = loss.item()
    assert final_loss < initial_loss, (initial_loss, final_loss)
    print(f"[train] loss {initial_loss:.3f} -> {final_loss:.3f}  OK")


def main() -> int:
    print("== voice_defense smoke tests ==")
    tests = [
        test_metrics,
        test_attack_zoo_dummy,
        test_protocol_loader_and_dataset,
        test_rawboost_runs,
        test_loss_math,
        test_aasist_backend_shape,
        test_full_pipeline_no_ssl,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    if failed:
        print(f"\n{failed}/{len(tests)} tests FAILED")
        return 1
    print(f"\nall {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
