import torch


def get_device() -> torch.device:
    """mps -> cuda -> cpu fallback, per CLAUDE.md dev environment notes."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
