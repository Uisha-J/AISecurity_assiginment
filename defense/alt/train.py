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
                sc.extend(torch.softmax(model(f.to(device)), 1)[:, 1].cpu().numpy())
                lb_all.extend(lb.numpy())
        val_eer = compute_eer(lb_all, sc)
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
