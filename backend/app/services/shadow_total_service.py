"""Validated shadow totals; the active production model is never changed here."""
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare_shadow_features(features, manifest):
    from ml.src.features.feature_builder import FEATURE_NAMES
    from ml.src.features.dataset_cache import CACHE_FORMAT_VERSION
    if manifest.get('feature_names') != FEATURE_NAMES or manifest.get('feature_semantic_version') != CACHE_FORMAT_VERSION:
        raise ValueError('Shadow preprocessing version/order mismatch')
    value=np.asarray(features,dtype=np.float64).reshape(1,-1).copy()
    if value.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(value).all():
        raise ValueError('Invalid shadow input vector')
    for name in manifest['disabled_features']:
        value[:,FEATURE_NAMES.index(name)]=0
    return value


def shadow_totals(features, *, as_of, lookback, target_day=None, root=None):
    from app.config import settings
    from ml.src.models.xgboost_model import XGBoostTrainer
    root=Path(root or settings.model_path).resolve()
    selector=root/'shadow_totals.json'
    if not selector.exists():
        return None
    if lookback!=10:
        return dict(status='unavailable',reason='影子模型目前仅验证了回溯10场',candidates=[])
    catalog=json.loads(selector.read_text(encoding='utf-8'))
    bundle=(root/catalog['bundle']).resolve()
    if not bundle.is_relative_to(root):
        raise ValueError('Shadow model directory outside model root')
    result=dict(status='diagnostic_shadow',version=catalog['version'],as_of=as_of.isoformat(),
                acceptance_run=False,candidates=[])
    for mode in ('direct','residual'):
        path=bundle/f'{mode}_manifest.json'
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=catalog['manifest_sha256'][mode]:
            raise ValueError('Shadow manifest changed')
        manifest=json.loads(raw)
        if str(target_day or as_of.date())<=manifest['training_data_until']:
            return dict(status='unavailable',reason='目标比赛处于影子模型训练范围内',candidates=[])
        for filename,expected in manifest['file_hashes'].items():
            if Path(filename).name!=filename:
                raise ValueError('Invalid shadow artifact name')
            artifact=bundle/'models'/mode/filename
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=expected:
                raise ValueError('Shadow artifact changed')
        value=prepare_shadow_features(features,manifest)
        model=XGBoostTrainer(models_dir=str(bundle/'models'))
        model.load(mode)
        predicted=float(model._predict_totals(value)[0])
        if not np.isfinite(predicted) or not 150<=predicted<=300:
            raise ValueError('Shadow total failed plausibility check')
        vector=value[0].tolist()
        result['candidates'].append(dict(strategy=mode,
            version=f"{catalog['version']}_{mode}",predicted_total=round(predicted,1),
            raw_predicted_total=predicted,training_data_until=manifest['training_data_until'],
            injury_adjustment_enabled=False,preprocessing_verified=True,
            processed_feature_vector=vector,
            feature_sha256=hashlib.sha256(value.astype('<f8').tobytes()).hexdigest()))
    return result
