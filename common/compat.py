"""Compatibility patches for PyTorch 2.7+ and SpeechBrain on Windows."""

import torch


def patch_torchaudio_soundfile():
    """Route torchaudio.load through soundfile to avoid the torchcodec/FFmpeg path.

    PyTorch 2.9 makes torchaudio.load dispatch to torchcodec, which needs FFmpeg
    *shared* libraries installed system-wide. On Windows that's painful, and
    Coqui XTTS calls torchaudio.load internally to read the speaker reference.
    soundfile reads wav/flac/ogg without FFmpeg, so we patch load to use it.
    """
    try:
        import torchaudio
        import soundfile as sf
        import numpy as np

        def _sf_load(filepath, *args, **kwargs):
            wav, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
            return torch.from_numpy(wav.T.copy()), sr  # (channels, time)

        torchaudio.load = _sf_load
    except Exception:
        pass


def patch_speechbrain_lazy():
    """Stop SpeechBrain lazy modules from doing a hard import on dunder access.

    PyTorch 2.9 registers custom ops and, while building a warning, calls
    inspect.getmodule which does ``hasattr(module, "__file__")`` over every entry
    in sys.modules. SpeechBrain registers LazyModules (e.g. integrations.k2_fsa)
    whose __getattr__ triggers a real import even for ``__file__`` — and k2 is not
    installable on Windows, so that import raises and crashes unrelated code
    (e.g. XTTS synthesis). Make dunder access on lazy modules raise AttributeError
    instead, so introspection skips them gracefully.
    """
    try:
        from speechbrain.utils import importutils as _iu
        _orig = _iu.LazyModule.__getattr__

        def _safe_getattr(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _orig(self, name)

        _iu.LazyModule.__getattr__ = _safe_getattr
    except Exception:
        pass


def patch_symlink_fallback_copy():
    """Make Path.symlink_to fall back to copy when symlinks aren't permitted.

    SpeechBrain 1.x links cached HF files into the savedir with
    ``dst.symlink_to(src)``, which raises WinError 1314 on Windows without admin
    / Developer Mode. Patching at the pathlib level guarantees the fallback no
    matter which SpeechBrain code path calls it, and only changes behaviour when
    the symlink actually fails.
    """
    try:
        import pathlib
        import shutil
        _orig_symlink = pathlib.Path.symlink_to

        def _symlink_or_copy(self, target, target_is_directory=False):
            try:
                return _orig_symlink(self, target, target_is_directory)
            except OSError:
                src = pathlib.Path(target)
                if src.is_dir():
                    shutil.copytree(str(src), str(self), dirs_exist_ok=True)
                else:
                    shutil.copy(str(src), str(self))

        pathlib.Path.symlink_to = _symlink_or_copy
    except Exception:
        pass


def apply_all_patches():
    _original = torch.load
    def _patched(*a, **kw):
        if "weights_only" not in kw: kw["weights_only"] = False
        return _original(*a, **kw)
    torch.load = _patched

    patch_torchaudio_soundfile()
    patch_speechbrain_lazy()
    patch_symlink_fallback_copy()

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
