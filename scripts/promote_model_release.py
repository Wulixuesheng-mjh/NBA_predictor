"""Validate and atomically promote a shadow model release."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path


def _validate_group(root: Path, version: str | None, hashes: dict | None) -> None:
    if version is None:
        return
    if Path(version).name!=version:
        raise ValueError('Invalid model version')
    folder = (root / version).resolve()
    if not folder.is_relative_to(root.resolve()) or not hashes:
        raise ValueError('Model path or file hashes missing/invalid')
    if not folder.is_dir():
        raise FileNotFoundError(f"missing model directory: {folder}")
    required=({'win_model.json','spread_model.json','total_model.json','metadata.json'}
              if root.name=='xgboost' else {'model.onnx','scaler.npz','metadata.json'})
    if not required.issubset(hashes):
        raise ValueError('Incomplete model file checksums')
    if root.name=='xgboost' and json.loads((folder/'metadata.json').read_text()).get('total_mode')=='residual':
        if 'total_direct_model.json' not in hashes:
            raise ValueError('Residual fallback model checksum missing')
    for name, expected in (hashes or {}).items():
        if Path(name).name!=name:
            raise ValueError('Invalid model artifact filename')
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(f"missing release file: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"hash mismatch: {path}")


def promote(models_dir: Path, version: str, evidence_path: Path | None = None) -> Path:
    if evidence_path is None:
        raise ValueError('Promotion requires offline, engineering and 200-game acceptance evidence')
    candidate = models_dir / "candidates" / f"{version}.json"
    manifest = json.loads(candidate.read_text(encoding="utf-8"))
    if str(manifest.get("version")) != version:
        raise ValueError("candidate version does not match manifest")
    file_hashes = manifest.get("file_hashes") or {}
    if not manifest.get('xgboost_version'):
        raise ValueError('Primary XGBoost bundle is required')
    _validate_group(models_dir / "xgboost", manifest.get("xgboost_version"), file_hashes.get("xgboost"))
    _validate_group(models_dir / "mlp", manifest.get("mlp_version"), file_hashes.get("mlp"))
    from app.services.release_gate_service import validate_promotion
    asyncio.run(validate_promotion(models_dir,version,evidence_path))

    active = models_dir / "active.json"
    models_dir.mkdir(parents=True, exist_ok=True)
    if active.exists():
        rollback = models_dir / "rollback.json"
        rollback_tmp = rollback.with_name(rollback.name + ".tmp")
        rollback_tmp.write_bytes(active.read_bytes())
        os.replace(rollback_tmp, rollback)
    tmp = active.with_name(active.name + ".tmp")
    tmp.write_bytes(candidate.read_bytes())
    os.replace(tmp, active)
    return active


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--models-dir", default=os.getenv("MODEL_PATH", "ml/saved_models"))
    parser.add_argument('--evidence',type=Path,required=True)
    args = parser.parse_args()
    print(promote(Path(args.models_dir), args.version,args.evidence))


if __name__ == "__main__":
    main()
