"""Compatibility patches for PyTorch 2.7+ and SpeechBrain on Windows."""

import torch


def apply_all_patches():
    _original = torch.load
    def _patched(*a, **kw):
        if "weights_only" not in kw: kw["weights_only"] = False
        return _original(*a, **kw)
    torch.load = _patched

    try:
        import importlib
        for mod in ("speechbrain.pretrained.fetching", "speechbrain.utils.fetching"):
            try:
                spec = importlib.util.find_spec(mod)
                if spec and spec.origin:
                    with open(spec.origin) as f: content = f.read()
                    old = "        destination.symlink_to(sourcepath)"
                    new = "        try:\n            destination.symlink_to(sourcepath)\n        except OSError:\n            import shutil as _s; _s.copy(str(sourcepath), str(destination))"
                    if old in content and "except OSError" not in content:
                        with open(spec.origin, "w") as f: f.write(content.replace(old, new))
            except (ImportError, AttributeError, TypeError): pass
    except Exception: pass
