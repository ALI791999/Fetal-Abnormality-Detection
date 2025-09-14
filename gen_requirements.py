#!/usr/bin/env python3
"""
Generate requirements.txt (and optional environment.yml) by scanning the repo's Python files.
- Detects imports via AST
- Excludes stdlib and local modules/packages
- Maps top-level modules to PyPI distributions and pins versions
"""

import os, ast, argparse, sys
from pathlib import Path
import importlib.util as ilu
import importlib.metadata as ilm

SITE_HINTS = ("site-packages", "dist-packages", "conda", "Frameworks/Python.framework")

def is_third_party(module: str) -> bool:
    """Heuristic: third-party if spec.origin lives in site/dist-packages (or conda paths)."""
    try:
        spec = ilu.find_spec(module)
    except (ModuleNotFoundError, ValueError):
        return False
    if spec is None:
        return False
    origin = getattr(spec, "origin", None)
    # builtins & namespace pkgs
    if origin in (None, "built-in"):
        return False
    # stdlib modules usually live in .../lib/pythonX.Y/...
    # third-party tend to live in site/dist-packages or conda envs
    path = str(origin)
    return any(h in path for h in SITE_HINTS)

def is_local_module(module: str, roots: list[Path]) -> bool:
    """Consider module local if a .py or package dir exists under repo roots."""
    candidates = []
    for r in roots:
        candidates += [
            r / f"{module}.py",
            r / module / "__init__.py",
        ]
    return any(p.exists() for p in candidates)

def discover_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if ".venv" not in str(p) and "venv" not in str(p)
            and "site-packages" not in str(p) and "dist-packages" not in str(p)
            and ".ipynb_checkpoints" not in str(p)]

def collect_top_imports(py_files: list[Path]) -> set[str]:
    top = set()
    for f in py_files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name:
                        top.add(name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    name = node.module.split(".")[0]
                    if name:
                        top.add(name)
    return top

def map_to_distributions(modules: set[str]) -> dict[str, list[str]]:
    # e.g. {"cv2": ["opencv-python"], "PIL": ["Pillow"], "sklearn": ["scikit-learn"]}
    try:
        mapping = ilm.packages_distributions()
    except Exception:
        mapping = {}
    return {m: mapping.get(m, []) for m in modules}

def resolve_requirements(modules: set[str], roots: list[Path]) -> list[str]:
    third_party = set()
    for m in modules:
        if is_local_module(m, roots):
            continue
        if is_third_party(m):
            third_party.add(m)

    mod_to_dists = map_to_distributions(third_party)

    # Fallbacks for uncommon names if mapping is missing but module is clearly 3rd-party
    manual = {
        "segmentation_models_pytorch": "segmentation-models-pytorch",
    }

    reqs = {}
    for m in sorted(third_party):
        dists = mod_to_dists.get(m, [])
        dist_name = None
        if dists:
            # pick the first distribution (usually correct)
            dist_name = dists[0]
        elif m in manual:
            dist_name = manual[m]
        else:
            # Last-resort: try the module name itself as a dist
            dist_name = m

        # Get version if installed
        try:
            ver = ilm.version(dist_name)
            reqs[dist_name] = ver
        except ilm.PackageNotFoundError:
            # leave unpinned if not found (maybe not installed yet)
            reqs.setdefault(dist_name, None)

    lines = []
    for name in sorted(reqs):
        v = reqs[name]
        if v:
            lines.append(f"{name}=={v}")
        else:
            lines.append(name)
    return lines

def write_requirements(lines: list[str], out: Path):
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")

def write_environment_yml(lines: list[str], out: Path, name: str):
    # Put everything under pip section; users with conda can `conda env create -f environment.yml`
    pip_block = "\n  - pip:\n" + "\n".join([f"    - {ln}" for ln in lines])
    yml = f"""name: {name}
channels:
  - conda-forge
  - defaults
dependencies:
  - python={sys.version_info.major}.{sys.version_info.minor}
{pip_block}
"""
    out.write_text(yml, encoding="utf-8")
    print(f"Wrote {out}")

def main():
    ap = argparse.ArgumentParser(description="Auto-generate requirements from project imports.")
    ap.add_argument("--root", default=".", help="Project root to scan")
    ap.add_argument("--out", default="requirements.txt", help="Output requirements.txt path")
    ap.add_argument("--env-yaml", default=None, help="Also write environment.yml to this path")
    ap.add_argument("--env-name", default="hc18-seg", help="Conda env name when writing environment.yml")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    py_files = discover_py_files(root)
    modules = collect_top_imports(py_files)
    req_lines = resolve_requirements(modules, [root])

    write_requirements(req_lines, Path(args.out))
    if args.env_yaml:
        write_environment_yml(req_lines, Path(args.env_yaml), args.env_name)

if __name__ == "__main__":
    main()
