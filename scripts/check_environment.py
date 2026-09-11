"""Read-only code/runtime checks; no training or data mutation."""
import ast
import importlib
import json
import platform
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
assert platform.system()=="Linux", "Run inside WSL2 Ubuntu"
expected=Path("/mnt/d/Projects/Scenario_Generation_Research")
assert root==expected, f"Unexpected workspace {root}"
paths=list((root/"scenario_lab").glob("*.py"))+list((root/"tests").glob("*.py"))+list(root.glob("*.py"))+list((root/"research_audit_20260910").glob("*.py"))
for path in paths:
    ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path))
versions={}
for name in ["torch","numpy","pandas","scipy","matplotlib","pytest","psutil","docx"]:
    module=importlib.import_module(name)
    versions[name]=str(getattr(module,"__version__","unknown"))
import torch
assert torch.cuda.is_available(), "CUDA unavailable in selected Python"
print(json.dumps(dict(workspace=str(root),platform=platform.platform(),python=sys.executable,
                     source_files_parsed=len(paths),dependencies=versions,
                     gpu=torch.cuda.get_device_name(0),data_available=(root/"Data/Waymo").is_dir()),indent=2))
