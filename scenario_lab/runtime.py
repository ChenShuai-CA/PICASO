"""Explicit runtime selection and provenance for Linux/WSL experiments."""
import json
import platform
import sys
from pathlib import Path
import torch


def resolve_device(requested="auto"):
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable; activate scenario-gpu")
    return str(selected)


def record_runtime(output, device):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    record = dict(platform=platform.platform(), python=sys.version.split()[0],
                  executable=sys.executable, cwd=str(Path.cwd()), torch=str(torch.__version__),
                  cuda_runtime=torch.version.cuda, device=str(device),
                  cuda_available=torch.cuda.is_available())
    if torch.device(device).type == "cuda":
        record["gpu"] = torch.cuda.get_device_name(device)
    (path / "runtime.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
