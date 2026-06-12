"""Train and fine-tune LCNN / RawNet2 on ASVspoof."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .models import LCNN, RawNet2
from .detector import load_alt_detector
from ...common.redteam_data import ASVspoofDataset, ClonedVoiceDataset
from ...evaluation.metrics import compute_eer

logger = logging.getLogger(__name__)


def train_alt_detector(data_root, output_root, model_type="lcnn", feature_type="lfcc",
                       n_lfcc=60, max_audio_len=64000, batch_size=32, epochs=30,
                       lr=1e-4, weight_decay=1e-4, early_stopping_patience=5, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    model_dir = Path(output_root) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    feat = feature_type if model_type == "lcnn" else "raw"

    train_ds = ASVspoofDataset(data_root, "train", feat, max_audio_len, n_lfcc)
    dev_ds = ASVspoofDataset(data_root, "dev", feat, max_audio_len, n_lfcc)
    w = train_ds.get_class_weights()
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          sampler=WeightedRandomSampler([w[s["label"]].item() for s in train_ds.samples], len(train_ds)), num_workers=4)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = (LCNN(n_lfcc) if model_type == "lcnn" else RawNet2()).to(device)
    criterion = nn.CrossEntropyLoss(weight=w.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_eer, patience = float("inf"), 0
    best_path = model_dir / f"{model_type}_best.pth"

    for epoch in range(epochs):
        model.train()
        loss_sum = sum(
            (optimizer.zero_grad(), (l := criterion(model(f.to(device)), lb.to(device))), l.backward(), optimizer.step(), l.item())[-1]
            for f, lb in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
        )
        model.eval()
        sc, lb_all = [], []
        with torch.no_grad():
            for f, lb in dev_dl:
                # Label convention: 1 == bonafide, 0 == spoof (matches evaluation/metrics.py).
                # Class index 1 == bonafide, so softmax[:, 1] == P(bonafide) (higher == bonafide).
                sc.extend(torch.softmax(model(f.to(device)), 1)[:, 1].cpu().numpy())
                lb_all.extend(lb.numpy())
        # compute_eer signature is (scores, labels) and returns (eer, threshold).
        val_eer, _ = compute_eer(sc, lb_all)
        scheduler.step()
        logger.info(f"Epoch {epoch+1}: Loss={loss_sum/len(train_dl):.4f} EER={val_eer:.4f}")

        if val_eer < best_eer:
            best_eer, patience = val_eer, 0
            torch.save({"model_state_dict": model.state_dict(), "model_type": model_type, "val_eer": val_eer}, best_path)
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    return {"best_eer": best_eer, "best_model_path": str(best_path)}


def _speaker_of(path) -> str:
    """Speaker id = parent directory name (cloned_audio/<spk>/clone_*.wav)."""
    return Path(path).parent.name


def train_cm_on_clones(real_paths, cloned_paths, output_root, model_type="lcnn",
                       feature_type="lfcc", n_lfcc=60, max_audio_len=64000,
                       batch_size=16, epochs=20, lr=1e-4, weight_decay=1e-4,
                       dev_speaker_frac=0.2, early_stopping_patience=5, seed=42,
                       device="cuda"):
    """Train an LCNN/RawNet2 countermeasure from scratch on real vs cloned audio.

    No ASVspoof needed: bonafide = real speaker audio (LibriSpeech references),
    spoof = XTTS clones. Speakers are split disjointly into train/dev to avoid
    leakage; if the speaker split is degenerate (too few speakers) it falls back
    to a file-level random split with a warning.

    Label convention (project-wide): 1 == bonafide (real), 0 == spoof (cloned).
    Saves a checkpoint compatible with ``load_alt_detector`` / ``run_asv_bypass``.
    """
    import random as _random

    device = device if torch.cuda.is_available() else "cpu"
    model_dir = Path(output_root) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    feat = feature_type if model_type == "lcnn" else "raw"

    if not real_paths or not cloned_paths:
        raise ValueError(
            f"Need both real and cloned audio (got {len(real_paths)} real, "
            f"{len(cloned_paths)} cloned). Run the attack/clone phase first."
        )

    # --- speaker-disjoint train/dev split (no leakage) ---
    rng = _random.Random(seed)
    speakers = sorted({_speaker_of(p) for p in real_paths}
                      | {_speaker_of(p) for p in cloned_paths})
    rng.shuffle(speakers)
    n_dev = max(1, int(len(speakers) * dev_speaker_frac))
    dev_spk = set(speakers[:n_dev])

    def _split(paths):
        tr = [p for p in paths if _speaker_of(p) not in dev_spk]
        dv = [p for p in paths if _speaker_of(p) in dev_spk]
        return tr, dv

    real_tr, real_dv = _split(real_paths)
    clone_tr, clone_dv = _split(cloned_paths)

    if not (real_tr and clone_tr and real_dv and clone_dv):
        logger.warning("Speaker-disjoint split degenerate (too few speakers); "
                       "falling back to file-level random split (possible leakage).")
        rng.shuffle(real_paths); rng.shuffle(cloned_paths)
        rc = max(1, int(len(real_paths) * dev_speaker_frac))
        cc = max(1, int(len(cloned_paths) * dev_speaker_frac))
        real_dv, real_tr = real_paths[:rc], real_paths[rc:]
        clone_dv, clone_tr = cloned_paths[:cc], cloned_paths[cc:]

    logger.info("CM-on-clones split: train real=%d clone=%d | dev real=%d clone=%d",
                len(real_tr), len(clone_tr), len(real_dv), len(clone_dv))

    train_ds = ClonedVoiceDataset(real_tr, clone_tr, feat, max_audio_len, n_lfcc)
    dev_ds = ClonedVoiceDataset(real_dv, clone_dv, feat, max_audio_len, n_lfcc)

    # class weights (clones usually outnumber references)
    labels = [s["label"] for s in train_ds.samples]
    n0, n1, t = labels.count(0), labels.count(1), len(labels)
    w = torch.tensor([t / (2 * max(n0, 1)), t / (2 * max(n1, 1))])
    sampler = WeightedRandomSampler([w[s["label"]].item() for s in train_ds.samples], t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = (LCNN(n_lfcc) if model_type == "lcnn" else RawNet2()).to(device)
    criterion = nn.CrossEntropyLoss(weight=w.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_eer, patience = float("inf"), 0
    best_path = model_dir / f"{model_type}_best.pth"

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for f, lb in tqdm(train_dl, desc=f"CM Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            loss = criterion(model(f.to(device)), lb.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        model.eval()
        sc, lb_all = [], []
        with torch.no_grad():
            for f, lb in dev_dl:
                # softmax[:, 1] == P(bonafide); higher == more bonafide.
                sc.extend(torch.softmax(model(f.to(device)), 1)[:, 1].cpu().numpy())
                lb_all.extend(lb.numpy())
        val_eer, _ = compute_eer(sc, lb_all)
        scheduler.step()
        logger.info(f"CM Epoch {epoch+1}: Loss={loss_sum/max(len(train_dl),1):.4f} dev_EER={val_eer:.4f}")

        # Save on <= so that when dev EER plateaus (e.g. hits 0.0 early on easy
        # data) the *latest, better-trained* model overwrites the early one.
        # Patience still increments unless EER strictly improves, so early
        # stopping continues to work.
        improved = val_eer < best_eer
        if val_eer <= best_eer:
            torch.save({"model_state_dict": model.state_dict(),
                        "model_type": model_type, "val_eer": val_eer}, best_path)
        if improved:
            best_eer, patience = val_eer, 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                logger.info("Early stopping (dev EER not improving).")
                break

    logger.info("CM-on-clones done. best dev EER=%.4f -> %s", best_eer, best_path)
    return {"best_eer": best_eer, "best_model_path": str(best_path)}


def finetune_alt_detector(checkpoint_path, real_paths, cloned_paths, output_root,
                          feature_type="lfcc", n_lfcc=60, max_audio_len=64000,
                          cloned_ratio=0.3, batch_size=32, epochs=10, lr=5e-5, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    model, mtype = load_alt_detector(checkpoint_path, device)
    n = int(len(cloned_paths) * cloned_ratio)
    feat = feature_type if mtype == "lcnn" else "raw"
    dl = DataLoader(ClonedVoiceDataset(real_paths[:n], cloned_paths[:n], feat, max_audio_len, n_lfcc),
                    batch_size=batch_size, shuffle=True, num_workers=4)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        loss_sum = sum(
            (opt.zero_grad(), (l := crit(model(f.to(device)), lb.to(device))), l.backward(), opt.step(), l.item())[-1]
            for f, lb in tqdm(dl, desc=f"Finetune {ep+1}/{epochs}")
        )
        logger.info(f"Finetune {ep+1}: Loss={loss_sum/len(dl):.4f}")
    ft_path = Path(output_root) / "models" / f"{mtype}_finetuned.pth"
    ft_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_type": mtype}, ft_path)
    return {"finetuned_model_path": str(ft_path)}
