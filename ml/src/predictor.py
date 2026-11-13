"""Prediction service — dual-model inference with consensus analysis.

Takes a feature vector and runs inference through both XGBoost and MLP,
producing a combined prediction with:
  - Individual model outputs
  - Consensus analysis (agree/disagree per target)
  - Calibrated outputs (if calibration is available)
  - Data quality warnings
  - Outputs rounded to nearest 0.5 (matching gambling line conventions)
"""

import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ml.src.config import ml_settings
from ml.src.features.feature_builder import FeatureVector, CoverOverFeatureVector
from ml.src.models.xgboost_model import XGBoostTrainer, XGBoostPrediction, CoverOverPrediction
from ml.src.models.calibration import Calibrator

# Lazy import MLP (requires torch — heavy dependency)
if TYPE_CHECKING:
    from ml.src.models.mlp_model import MLPTrainer, MLPPrediction

logger = logging.getLogger(__name__)


def round_to_half(value: float) -> float:
    """Round a value to the nearest 0.5.

    Basketball scores are integers, and gambling lines use .5 endings
    (e.g. +2.5, -3.5, 220.5, 223.5) to avoid pushes. Rounding model
    outputs to 0.5 makes predictions directly comparable to betting lines.
    """
    return round(value * 2) / 2


@dataclass
class ConsensusAnalysis:
    """Analysis of agreement between XGBoost and MLP."""

    win_agree: bool
    spread_direction_agree: bool  # Both predict same sign (home win/loss margin)
    total_agree: bool  # Both agree on over/under vs league average
    agreement_score: float  # 0.0 - 1.0: how well models agree overall


@dataclass
class ConfidenceLevel:
    """Prediction confidence assessment."""

    level: str  # "high" | "medium" | "low" | "not_recommended"
    score: float  # 0-100 numeric confidence score
    factors: dict[str, float]  # breakdown of contributing factors
    label_cn: str  # Chinese label for frontend display


@dataclass
class CombinedPrediction:
    """Complete prediction combining both models."""

    home_team_id: int
    away_team_id: int
    lookback_games: int
    actual_games_home: int
    actual_games_away: int
    predicted_at: str

    xgboost: XGBoostPrediction | None
    mlp: "MLPPrediction | None"
    consensus: ConsensusAnalysis | None
    recommended: str  # "xgboost" (primary per plan)

    data_quality: dict
    calibration_applied: bool = False
    confidence: ConfidenceLevel | None = None
    raw_outputs: dict = field(default_factory=dict)
    # Kept out of the public response; persisted per model in the audit ledger.
    processed_inputs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to API response format."""
        result: dict = {
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "lookback_games": self.lookback_games,
            "actual_games_home": self.actual_games_home,
            "actual_games_away": self.actual_games_away,
            "predicted_at": self.predicted_at,
            "recommended": self.recommended,
            "calibration_applied": self.calibration_applied,
            "data_quality": self.data_quality,
            "raw_outputs": self.raw_outputs,
        }

        if self.xgboost:
            result["xgboost"] = {
                "model_version": self.xgboost.model_version,
                "win_prob_home": self.xgboost.win_prob_home,
                "win_prob_away": self.xgboost.win_prob_away,
                "predicted_spread": self.xgboost.predicted_spread,
                "spread_confidence": self.xgboost.spread_confidence,
                "predicted_total": self.xgboost.predicted_total,
                "total_confidence": self.xgboost.total_confidence,
            }

        if self.mlp:
            result["mlp"] = {
                "model_version": self.mlp.model_version,
                "win_prob_home": self.mlp.win_prob_home,
                "win_prob_away": self.mlp.win_prob_away,
                "predicted_spread": self.mlp.predicted_spread,
                "spread_confidence": self.mlp.spread_confidence,
                "predicted_total": self.mlp.predicted_total,
                "total_confidence": self.mlp.total_confidence,
            }

        if self.consensus:
            result["consensus"] = {
                "win_agree": self.consensus.win_agree,
                "spread_direction_agree": self.consensus.spread_direction_agree,
                "total_agree": self.consensus.total_agree,
                "agreement_score": self.consensus.agreement_score,
            }

        if self.confidence:
            result["confidence"] = {
                "level": self.confidence.level,
                "score": round(self.confidence.score, 1),
                "factors": self.confidence.factors,
                "label_cn": self.confidence.label_cn,
            }

        return result


class Predictor:
    """Unified prediction service using both XGBoost and MLP."""

    def __init__(
        self,
        xgb_trainer: XGBoostTrainer | None = None,
        mlp_trainer: "MLPTrainer | None" = None,
        calibrator: Calibrator | None = None,
        release_manifest: dict | None = None,
    ):
        self.xgb_trainer = xgb_trainer
        self.mlp_trainer = mlp_trainer
        self.calibrator = calibrator or Calibrator()
        self.release_manifest = deepcopy(release_manifest or {})
        self._shap_explainer = None  # Lazy-initialized SHAP TreeExplainer (win)
        self._shap_explainer_cover = None  # Lazy-initialized SHAP TreeExplainer (cover)
        self._shap_explainer_over = None  # Lazy-initialized SHAP TreeExplainer (over)

    def predict(
        self,
        feature_vector: FeatureVector,
        apply_calibration: bool = True,
    ) -> CombinedPrediction:
        """Generate prediction from a feature vector.

        Args:
            feature_vector: Complete FeatureVector from FeatureBuilder.
            apply_calibration: Whether to apply calibration coefficients.

        Returns:
            CombinedPrediction with both model outputs and consensus.
        """
        from ml.src.features.feature_builder import FEATURE_NAMES
        if feature_vector.feature_names != FEATURE_NAMES:
            raise ValueError('Builder feature order mismatch')
        X = feature_vector.features.reshape(1, -1)

        extra_warnings: list[str] = []
        raw_outputs = {}
        processed_inputs = {}
        X_for_release = self._prepare_release_features(X)
        if self._release_semantic_version() < 6:
            extra_warnings.append(
                "当前活动模型使用旧特征语义；已应用可兼容的量纲转换，建议完成新版本影子验收"
            )

        # Calibration is opt-in: the variance-inflating variant made MAE worse
        # on the validation set, so by default we serve raw model outputs.
        # Coefficients are fitted on XGBoost only — never applied to the MLP.
        use_calibration = bool(
            apply_calibration
            and getattr(ml_settings, "calibration_enabled", False)
            and self.release_manifest.get("calibration_enabled") is True
            and self.calibrator.is_fitted
        )

        # XGBoost prediction (primary)
        xgb_pred = None
        if self.xgb_trainer and self.xgb_trainer.is_trained:
            try:
                expected = int(self.xgb_trainer.model_win.n_features_in_)
                model_input = self._model_features(X_for_release, expected)
                processed_inputs['xgboost'] = self._input_snapshot(
                    model_input, 'xgboost', self.xgb_trainer.version)
                result = self.xgb_trainer.predict(model_input)
                # xgb predict returns list for 2D input, single object for 1D
                xgb_pred = result[0] if isinstance(result, list) else result
                raw_outputs["xgboost"] = asdict(xgb_pred)
                if use_calibration:
                    xgb_pred = self.calibrator.apply(xgb_pred)
                # Round to nearest 0.5 — basketball scores are integers,
                # gambling lines use .5 to avoid pushes
                xgb_pred.predicted_spread = round_to_half(xgb_pred.predicted_spread)
                xgb_pred.predicted_total = round_to_half(xgb_pred.predicted_total)
                ok, reason = self._is_plausible(xgb_pred)
                if not ok:
                    logger.warning("XGBoost output rejected: %s", reason)
                    extra_warnings.append(f"XGBoost 输出异常（{reason}），已忽略")
                    xgb_pred = None
            except Exception as e:
                logger.warning("XGBoost prediction failed: %s", e)
                extra_warnings.append(f"XGBoost 推理失败：{e}")

        # MLP prediction (auxiliary) — check both PyTorch and ONNX backends
        mlp_pred = None
        mlp_available = (
            self.mlp_trainer
            and (self.mlp_trainer.model is not None or self.mlp_trainer._onnx_session is not None)
        )
        if mlp_available:
            try:
                expected = (
                    len(self.mlp_trainer._scaler_mean)
                    if self.mlp_trainer._scaler_mean is not None else X.shape[1]
                )
                model_input = self._model_features(X_for_release, expected)
                processed_inputs['mlp'] = self._input_snapshot(model_input, 'mlp', self.mlp_trainer.version)
                result = self.mlp_trainer.predict(model_input)
                mlp_pred = result[0] if isinstance(result, list) else result
                raw_outputs["mlp"] = asdict(mlp_pred)
                # NOTE: no calibration here. The coefficients are fitted on
                # XGBoost residuals and made MLP worse (e.g. totals pinned to
                # the 180 floor).
                # Round to nearest 0.5
                mlp_pred.predicted_spread = round_to_half(mlp_pred.predicted_spread)
                mlp_pred.predicted_total = round_to_half(mlp_pred.predicted_total)
                ok, reason = self._is_plausible(mlp_pred)
                if not ok:
                    logger.warning("MLP output rejected: %s", reason)
                    extra_warnings.append(f"MLP 输出异常（{reason}），已忽略该模型")
                    mlp_pred = None
            except Exception as e:
                logger.warning("MLP prediction failed: %s", e)
                extra_warnings.append(f"MLP 推理失败：{e}")

        # Consensus analysis
        consensus = None
        if xgb_pred and mlp_pred:
            consensus = self._analyze_consensus(xgb_pred, mlp_pred)

        # Data quality assessment
        data_quality = {
            "home_games_available": feature_vector.home_games_available,
            "away_games_available": feature_vector.away_games_available,
            "warnings": list(feature_vector.data_quality_warnings or []) + extra_warnings,
        }
        compatibility = getattr(self.mlp_trainer, "input_compatibility_id", None)
        if compatibility:
            data_quality["mlp_input_compatibility"] = compatibility

        # Determine recommendation
        if xgb_pred:
            recommended = "xgboost"
        elif mlp_pred:
            recommended = "mlp"
        else:
            recommended = "unavailable"

        # Compute overall confidence level
        confidence = self._compute_confidence(data_quality, xgb_pred, consensus)

        return CombinedPrediction(
            home_team_id=feature_vector.home_team_id,
            away_team_id=feature_vector.away_team_id,
            lookback_games=feature_vector.lookback_requested,
            actual_games_home=feature_vector.home_games_available,
            actual_games_away=feature_vector.away_games_available,
            predicted_at=datetime.now().isoformat(),
            xgboost=xgb_pred,
            mlp=mlp_pred,
            consensus=consensus,
            recommended=recommended,
            data_quality=data_quality,
            calibration_applied=use_calibration,
            confidence=confidence,
            raw_outputs=raw_outputs,
            processed_inputs=processed_inputs,
        )

    @staticmethod
    def _adapt_features(X: np.ndarray, expected_dim: int) -> np.ndarray:
        """Serve a previous release while new features are appended.

        Schema v6 guarantees that the legacy 166 columns remain an exact
        prefix.  Shrinking is therefore safe; expanding is never fabricated.
        """
        if X.shape[1] == expected_dim:
            return X
        if X.shape[1] > expected_dim:
            return X[:, :expected_dim]
        raise ValueError(f"Feature vector has {X.shape[1]} columns; model needs {expected_dim}")

    def _release_semantic_version(self) -> int:
        try:
            return int(self.release_manifest.get("feature_semantic_version") or 5)
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid release feature semantic version') from exc

    def _prepare_release_features(self, X: np.ndarray) -> np.ndarray:
        """Translate current features for an older pinned model when possible."""
        prepared = np.asarray(X, dtype=np.float64).copy()
        from ml.src.features.feature_builder import FEATURE_NAMES
        from ml.src.features.dataset_cache import CACHE_FORMAT_VERSION

        if prepared.ndim != 2 or prepared.shape[1] != len(FEATURE_NAMES) or not np.isfinite(prepared).all():
            raise ValueError('Invalid builder feature vector')
        semantic = self._release_semantic_version()
        if semantic not in (5, CACHE_FORMAT_VERSION):
            raise ValueError('Release feature semantics are incompatible with this code image')
        names = self.release_manifest.get('feature_names')
        if semantic >= 6 and (names != FEATURE_NAMES
                or self.release_manifest.get('feature_dim') != len(FEATURE_NAMES)):
            raise ValueError('Release feature dimension/order mismatch')
        if names is not None and names != FEATURE_NAMES[:len(names)]:
            raise ValueError('Legacy release feature order mismatch')
        disabled = self.release_manifest.get('disabled_features', [])
        if (not isinstance(disabled, list) or any(not isinstance(name, str) for name in disabled)
                or len(disabled) != len(set(disabled)) or any(name not in FEATURE_NAMES for name in disabled)):
            raise ValueError('Invalid disabled feature list')
        if self._release_semantic_version() < 6:
            # v5 trained expected_total with an erroneous /200 scale. Keep the
            # active v5 bundle stable while v6 remains in shadow evaluation.
            index = FEATURE_NAMES.index("expected_total")
            if index < prepared.shape[1]:
                prepared[:, index] /= 2.0
        for name in disabled:
            prepared[:, FEATURE_NAMES.index(name)] = 0.0
        return prepared

    def _model_features(self, X: np.ndarray, expected_dim: int) -> np.ndarray:
        declared = self.release_manifest.get('feature_dim')
        if declared is not None and declared != expected_dim:
            raise ValueError('Model dimension differs from its release manifest')
        # Only the documented legacy 166-column prefix may be truncated.
        if X.shape[1] != expected_dim and expected_dim != 166:
            raise ValueError('Undocumented feature truncation is forbidden')
        return self._adapt_features(X, expected_dim)

    def _input_snapshot(self, X, model_name, model_version):
        from ml.src.inference_snapshot import input_snapshot
        from ml.src.features.feature_builder import FEATURE_NAMES
        return input_snapshot(X[0], FEATURE_NAMES[:X.shape[1]], model_name=model_name,
                              model_version=model_version, manifest=self.release_manifest)

    @staticmethod
    def _is_plausible(pred) -> tuple[bool, str]:
        """Reject physically impossible model outputs.

        A broken network can emit a total of 5 or a 600-point spread; the UI
        must never present that as a real prediction (it previously surfaced as
        a clipped "180" total).
        """
        total = getattr(pred, "predicted_total", None)
        if total is not None:
            lo = getattr(ml_settings, "output_total_min", 150.0)
            hi = getattr(ml_settings, "output_total_max", 300.0)
            if not (lo <= total <= hi):
                return False, f"总分 {total:.1f} 超出 [{lo:.0f}, {hi:.0f}]"

        spread = getattr(pred, "predicted_spread", None)
        if spread is not None:
            cap = getattr(ml_settings, "output_max_abs_spread", 45.0)
            if abs(spread) > cap:
                return False, f"让分 {spread:.1f} 绝对值超过 {cap:.0f}"

        prob = getattr(pred, "win_prob_home", None)
        if prob is None or not (0.0 <= prob <= 1.0) or np.isnan(prob):
            return False, f"主胜概率 {prob} 非法"

        return True, ""

    @staticmethod
    def _compute_confidence(
        data_quality: dict,
        xgb_pred: XGBoostPrediction | None,
        consensus: ConsensusAnalysis | None,
    ) -> ConfidenceLevel:
        """Compute overall prediction confidence from multiple signals.

        Combines:
          1. Data sufficiency (games available) — gatekeeper
          2. Win probability decisiveness (distance from 0.5)
          3. Model consensus (if both models available)
          4. Spread/total confidence from XGBoost

        Returns a ConfidenceLevel with level, score, factors breakdown, and
        Chinese label for frontend display.
        """
        MIN_GAMES = 5
        home_games = data_quality.get("home_games_available", 0)
        away_games = data_quality.get("away_games_available", 0)
        min_games = min(home_games, away_games)

        # ── Factor 1: Data sufficiency (0-30 points) ──
        if min_games >= 10:
            data_score = 30.0
        elif min_games >= MIN_GAMES:
            data_score = 15.0 + (min_games - MIN_GAMES) * 3.0  # 15-30
        elif min_games >= 3:
            data_score = 5.0 + (min_games - 3) * 5.0  # 5-15
        else:
            data_score = max(0.0, min_games * 1.67)  # 0-5

        # If either team has < MIN_GAMES, hard-cap the confidence
        insufficient_data = min_games < MIN_GAMES

        # ── Factor 2: Win probability decisiveness (0-30 points) ──
        if xgb_pred:
            win_margin = abs(xgb_pred.win_prob_home - 0.5)  # 0.0 - 0.5
            decisiveness_score = win_margin * 60.0  # 0-30 points
        else:
            decisiveness_score = 0.0

        # ── Factor 3: Model consensus (0-20 points) ──
        if consensus:
            consensus_score = consensus.agreement_score * 20.0
        else:
            consensus_score = 0.0  # only one model available

        # ── Factor 4: Spread/total confidence from XGBoost (0-20 points) ──
        if xgb_pred:
            spread_conf = xgb_pred.spread_confidence or 0.0
            total_conf = xgb_pred.total_confidence or 0.0
            model_confidence_score = ((spread_conf + total_conf) / 2.0) * 20.0
        else:
            model_confidence_score = 0.0

        factors = {
            "data_sufficiency": round(data_score, 1),
            "win_decisiveness": round(decisiveness_score, 1),
            "model_consensus": round(consensus_score, 1),
            "model_confidence": round(model_confidence_score, 1),
        }

        total_score = data_score + decisiveness_score + consensus_score + model_confidence_score

        # ── Determine level ──
        if insufficient_data:
            level = "not_recommended"
            label_cn = "不建议参考"
        elif total_score >= 70:
            level = "high"
            label_cn = "高置信度"
        elif total_score >= 45:
            level = "medium"
            label_cn = "中等置信度"
        elif total_score >= 20:
            level = "low"
            label_cn = "低置信度"
        else:
            level = "not_recommended"
            label_cn = "不建议参考"

        return ConfidenceLevel(
            level=level,
            score=min(total_score, 100.0),
            factors=factors,
            label_cn=label_cn,
        )

    @staticmethod
    def _analyze_consensus(
        xgb: XGBoostPrediction, mlp: "MLPPrediction"
    ) -> ConsensusAnalysis:
        """Determine how much the two models agree."""
        # Win agreement: both predict same winner
        win_agree = (xgb.win_prob_home > 0.5) == (mlp.win_prob_home > 0.5)

        # Spread direction: both predict same sign (home wins or loses)
        spread_dir_agree = (xgb.predicted_spread > 0) == (mlp.predicted_spread > 0)

        # Spread within 2.5 points: models agree on magnitude
        spread_close = abs(xgb.predicted_spread - mlp.predicted_spread) <= 2.5

        # Total agreement: both within 5 points of each other AND
        # both on same side of 227.5 (rough NBA league average 2024-25)
        league_avg_total = 227.5
        total_agree = (
            abs(xgb.predicted_total - mlp.predicted_total) <= 5.0
            and (xgb.predicted_total > league_avg_total) == (mlp.predicted_total > league_avg_total)
        )

        # Overall agreement score (4 checks, equally weighted)
        agreement_score = (
            float(win_agree) + float(spread_dir_agree) + float(spread_close) + float(total_agree)
        ) / 4.0

        return ConsensusAnalysis(
            win_agree=win_agree,
            spread_direction_agree=spread_dir_agree,
            total_agree=total_agree,
            agreement_score=agreement_score,
        )


    def explain_prediction(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
        top_k: int = 5,
        display_names: dict[str, str] | None = None,
    ) -> list[dict] | None:
        """Explain the XGBoost win prediction using SHAP feature contributions.

        Uses a lazy-initialized TreeExplainer to compute Shapley values for
        the input feature vector, returning the top-K most influential features.

        Args:
            X: Feature vector, shape (1, FEATURE_DIM).
            feature_names: Optional list of feature names for annotation.
            top_k: Number of top contributing features to return.
            display_names: Optional dict mapping feature_name → Chinese label.

        Returns:
            List of dicts with feature, display_name, shap_value, contribution,
            and direction — or None if SHAP is unavailable.
        """
        if not self.xgb_trainer or not self.xgb_trainer.is_trained:
            return None
        if self.xgb_trainer.model_win is None:
            return None

        try:
            import shap

            expected = int(self.xgb_trainer.model_win.n_features_in_)
            X = self._adapt_features(self._prepare_release_features(X), expected)
            if feature_names is not None:
                feature_names = feature_names[:expected]

            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(
                    self.xgb_trainer.model_win,
                    feature_perturbation="tree_path_dependent",
                )

            shap_values = self._shap_explainer.shap_values(X)
            # For binary classifier: values is (n_samples, n_features, 2) or (n_samples, n_features)
            if hasattr(shap_values, 'values'):
                sv = shap_values.values
            else:
                sv = shap_values

            if hasattr(sv, 'ndim') and sv.ndim == 3:
                # shape (1, n_features, 2) → take class 1 (home win)
                contributions = sv[0, :, 1]
                base_val = float(shap_values.base_values[0, 1]) if hasattr(shap_values, 'base_values') else 0.5
            else:
                contributions = sv[0] if hasattr(sv, 'ndim') else sv
                base_val = float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0.5

            names = feature_names or [f"feature_{i}" for i in range(len(contributions))]
            sorted_contribs = sorted(
                zip(names, contributions), key=lambda x: abs(float(x[1])), reverse=True
            )

            result: list[dict] = []
            for fname, fval in sorted_contribs[:top_k]:
                fval_f = float(fval)
                display = (display_names or {}).get(fname, fname)
                result.append({
                    "feature": fname,
                    "display_name": display,
                    "shap_value": round(fval_f, 4),
                    "contribution": "positive" if fval_f > 0 else "negative",
                    "direction": "favors_home" if fval_f > 0 else "favors_away",
                })

            # Add base value for context
            return result

        except Exception:
            logger.debug("SHAP explanation unavailable", exc_info=True)
            return None


    def predict_cover_over(
        self,
        cover_over_vector: CoverOverFeatureVector,
        spread_line: float | None = None,
        total_line: float | None = None,
    ) -> dict | None:
        """Run 盘口 binary inference from the reduced cover/over feature vector.

        Returns a dict with direction + probability for the two binary bets:
          - 让分胜负 (cover): whether the home team covers the spread
          - 大小分 (over/under): whether the game goes over the total

        Returns None if the cover/over models are not trained/loaded.
        """
        if not self.xgb_trainer or not self.xgb_trainer.is_cover_over_trained:
            return None
        try:
            cov = self.xgb_trainer.predict_cover_over(cover_over_vector.features)
        except Exception as e:
            logger.warning("Cover/over prediction failed: %s", e)
            return None

        cover_home = cov.cover_prob_home >= 0.5
        over = cov.over_prob >= 0.5
        return {
            "model_version": cov.model_version,
            "spread_line": spread_line,
            "total_line": total_line,
            "cover_prob_home": round(cov.cover_prob_home, 4),
            "cover_prob_away": round(1.0 - cov.cover_prob_home, 4),
            "cover_direction": "home" if cover_home else "away",
            "cover_direction_cn": "主队赢盘" if cover_home else "客队赢盘",
            "over_prob": round(cov.over_prob, 4),
            "under_prob": round(1.0 - cov.over_prob, 4),
            "over_direction": "over" if over else "under",
            "over_direction_cn": "大分" if over else "小分",
        }

    def explain_cover_over(
        self,
        cover_over_vector: CoverOverFeatureVector,
        feature_names: list[str] | None = None,
        top_k: int = 5,
        display_names: dict[str, str] | None = None,
    ) -> dict | None:
        """Explain the cover and over XGBoost classifiers with SHAP.

        Returns {"cover": [...], "over": [...]} where each value is a list of
        top-K feature contributions (or None if SHAP is unavailable).
        """
        if not self.xgb_trainer or not self.xgb_trainer.is_cover_over_trained:
            return None
        try:
            import shap
        except ImportError:
            return None

        X = cover_over_vector.features.reshape(1, -1)
        names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        def _contributions(model, explainer_attr, pos_label, neg_label):
            explainer = getattr(self, explainer_attr)
            if explainer is None:
                explainer = shap.TreeExplainer(
                    model, feature_perturbation="tree_path_dependent"
                )
                setattr(self, explainer_attr, explainer)
            sv = explainer.shap_values(X)
            if hasattr(sv, "values"):
                sv = sv.values
            if hasattr(sv, "ndim") and sv.ndim == 3:
                contribs = sv[0, :, 1]  # class 1
            else:
                contribs = sv[0] if hasattr(sv, "ndim") else sv
            sorted_contribs = sorted(
                zip(names, contribs), key=lambda x: abs(float(x[1])), reverse=True
            )
            out: list[dict] = []
            for fname, fval in sorted_contribs[:top_k]:
                fval_f = float(fval)
                out.append({
                    "feature": fname,
                    "display_name": (display_names or {}).get(fname, fname),
                    "shap_value": round(fval_f, 4),
                    "contribution": "positive" if fval_f > 0 else "negative",
                    "direction": pos_label if fval_f > 0 else neg_label,
                })
            return out

        result: dict = {"cover": None, "over": None}
        try:
            result["cover"] = _contributions(
                self.xgb_trainer.model_cover,
                "_shap_explainer_cover",
                "favors_home_cover",
                "favors_away_cover",
            )
        except Exception:
            logger.debug("Cover SHAP unavailable", exc_info=True)
        try:
            result["over"] = _contributions(
                self.xgb_trainer.model_over,
                "_shap_explainer_over",
                "favors_over",
                "favors_under",
            )
        except Exception:
            logger.debug("Over SHAP unavailable", exc_info=True)

        return result if (result.get("cover") or result.get("over")) else None


def _read_active_release() -> dict | None:
    """Read the atomically published release manifest, if any."""
    import json
    import os

    path = os.path.join(ml_settings.models_dir, "active.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read active release manifest: %s", e)
        return None


def load_predictor(
    xgb_version: str | None = None,
    mlp_version: str | None = None,
    release_manifest: dict | None = None,
    mlp_compatibility_path: str | None = None,
) -> Predictor:
    """Load trained models and create a Predictor instance.

    Versions are pinned by ``saved_models/active.json`` when it exists, so a
    prediction never mixes a freshly written model with a stale scaler or an
    older MLP. ``mlp_version: null`` in that manifest means "MLP failed its
    checks — do not load any MLP" (rather than silently falling back to the
    newest old one).

    Args:
        xgb_version: XGBoost model version to load (release pin if None).
        mlp_version: MLP model version to load (release pin if None).
        mlp_compatibility_path: Explicit, hash-checked legacy input repair contract.
            Archived manifests carry the same contract for reproducible replay.

    Returns:
        Predictor ready for inference.
    """
    release = dict(release_manifest) if release_manifest is not None else _read_active_release()
    requested_version = xgb_version or mlp_version
    if requested_version is not None:
        # Explicit candidate/replay loads must carry their own preprocessing
        # semantics, rather than inheriting the currently active old bundle.
        if release_manifest is None and (not release or requested_version not in (
            release.get("xgboost_version"), release.get("mlp_version")
        )):
            import json
            from pathlib import Path

            path = Path(ml_settings.models_dir) / "candidates" / f"{requested_version}.json"
            release = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if release:
            for supplied, key in ((xgb_version, "xgboost_version"), (mlp_version, "mlp_version")):
                if supplied is not None and supplied != release.get(key):
                    raise ValueError("Requested models do not belong to one release bundle")
        elif xgb_version is None:
            xgb_version = requested_version
    skip_mlp = requested_version is not None and release is None and mlp_version is None
    if release:
        if xgb_version is None:
            xgb_version = release.get("xgboost_version")
        if mlp_version is None:
            pinned = release.get("mlp_version")
            mlp_version = pinned
            skip_mlp = "mlp_version" in release and pinned is None
        logger.info(
            "Active release %s: xgb=%s mlp=%s (data until %s)",
            release.get("version"),
            release.get("xgboost_version"),
            release.get("mlp_version"),
            release.get("training_data_until"),
        )

    xgb = XGBoostTrainer()
    try:
        xgb.load(xgb_version)
    except Exception as e:
        logger.warning("Could not load XGBoost model: %s", e)

    mlp = None
    if skip_mlp:
        logger.warning("MLP disabled by release manifest — XGBoost only")
    else:
        try:
            from ml.src.models.mlp_model import MLPTrainer
            mlp = MLPTrainer()
            try:
                mlp.load(mlp_version)
            except Exception as e:
                logger.warning("Could not load MLP model: %s", e)
                mlp = None
        except ImportError as e:
            logger.info("MLP not available (%s) — using XGBoost only", e)

    compatibility = (release or {}).get("mlp_input_compatibility")
    if mlp_compatibility_path is not None or compatibility is not None:
        if mlp is None:
            raise ValueError("Requested legacy MLP compatibility requires its pinned model")
        from ml.src.legacy_mlp_compatibility import CompatibleLegacyMLP
        mlp = CompatibleLegacyMLP(mlp, mlp_compatibility_path if mlp_compatibility_path is not None else compatibility)
        if compatibility is not None and compatibility != mlp.input_compatibility_contract:
            raise ValueError("Archived MLP compatibility differs from requested contract")
        release = dict(release or {})
        release["mlp_input_compatibility"] = mlp.input_compatibility_contract

    calibrator = Calibrator()
    try:
        calibrator.load()
    except Exception:
        logger.info("No calibration coefficients found, using raw predictions")

    return Predictor(xgb, mlp, calibrator, release_manifest=release)
