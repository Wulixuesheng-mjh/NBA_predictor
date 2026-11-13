"""Prepare one fixed repair and serving-source overlay; no model search."""
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ml.src.legacy_mlp_compatibility import contract_spec


def main():
    root=Path(__file__).resolve().parents[1];out=root/'reports/mlp_constant_repair_20261110'
    if out.exists():raise ValueError('Immutable repair experiment already prepared')
    out.mkdir(parents=True)
    # Use only the frozen serving predictor, never replace it with unrelated workspace changes.
    original=root/'tmp/total_root_cause_20261109/frozen_ml_src/predictor.py'
    data=original.read_bytes()
    if hashlib.sha256(data).hexdigest()!='52aaec35fab0d9b505b39835b8a8295c2eefb8bd2d83dc42cd0ad648c9558235':
        raise ValueError('Frozen predictor source changed')
    source=data.decode('utf-8')
    edits=[('    release_manifest: dict | None = None,\n) -> Predictor:',
        '    release_manifest: dict | None = None,\n    mlp_compatibility_path: str | None = None,\n) -> Predictor:'),
        ('    calibrator = Calibrator()\n',
         '    if mlp_compatibility_path is not None:\n        if mlp is None:\n            raise ValueError("Requested legacy MLP compatibility requires its pinned model")\n        from ml.src.legacy_mlp_compatibility import CompatibleLegacyMLP\n        mlp = CompatibleLegacyMLP(mlp, mlp_compatibility_path)\n\n    calibrator = Calibrator()\n'),
        ('        # Determine recommendation\n',
         '        compatibility = getattr(self.mlp_trainer, "input_compatibility_id", None)\n        if compatibility:\n            data_quality["mlp_input_compatibility"] = compatibility\n\n        # Determine recommendation\n')]
    for old,new in edits:
        if old=='    calibrator = Calibrator()\n':
            current=(root/'ml/src/predictor.py').read_text(encoding='utf-8')
            start=current.index('    compatibility = (release or {}).get("mlp_input_compatibility")')
            stop=current.index('    calibrator = Calibrator()',start)
            new=current[start:stop]+old
        if source.count(old)!=1:raise ValueError('Ambiguous frozen-source overlay')
        source=source.replace(old,new)
    (out/'frozen_predictor_repair.py').write_text(source,encoding='utf-8')
    (out/'contract.json').write_text(json.dumps(contract_spec(),indent=2),encoding='utf-8')
    (out/'protocol.json').write_text(json.dumps(dict(id='mlp_constant_repair_20261110_v1',
        selection='fixed evidence-based repair; no hyperparameter or model search',
        contract=contract_spec(),source_predictor_sha256=hashlib.sha256(data).hexdigest(),
        validation=['132 saved reconstructed inputs, all heads and XGB controls',
            '221 older probes; remaining guard rejections explicitly reported',
            '100 original training-cache inputs unchanged',
            'ONNX/PyTorch and single/batch all heads',
            'strict artifact/scaler/input contract negative tests'],
        labels='retrospective diagnosis only, not real pregame accuracy or promotion evidence',
        weight_training=False,primary_promotion=False),indent=2),encoding='utf-8')
    print(str(out))


if __name__=='__main__':main()
