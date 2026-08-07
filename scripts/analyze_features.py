"""Feature importance analysis and pruning tool.

Loads a trained XGBoost model, analyzes feature importance, and
identifies candidates for pruning (low importance + redundant features).

Usage:
    python scripts/analyze_features.py [--version 20260731_073446] [--top 20]

Outputs:
    1. Feature importance ranking (sorted by gain)
    2. Correlation matrix for top features
    3. Prune candidates (importance < 0.1% of total + high correlation)

The analysis is saved to ml/saved_models/xgboost/<version>/feature_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.config import ml_settings


def compute_correlation_matrix(
    features: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.92,
) -> list[dict[str, Any]]:
    """Identify highly correlated feature pairs.

    Returns pairs with |Pearson r| > threshold, sorted by correlation strength.
    """
    n = features.shape[1]
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            # Skip if either column is constant
            if np.std(features[:, i]) == 0 or np.std(features[:, j]) == 0:
                continue
            r = np.corrcoef(features[:, i], features[:, j])[0, 1]
            if abs(r) > threshold:
                pairs.append({
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "pearson_r": round(float(r), 4),
                })

    return sorted(pairs, key=lambda p: -abs(p["pearson_r"]))


def find_prune_candidates(
    importance: list[tuple[str, float]],
    correlation_pairs: list[dict],
    min_importance_pct: float = 0.1,
) -> dict[str, Any]:
    """Identify features that are safe to remove.

    Two categories of prune candidates:

    A) LOW IMPORTANCE — features with importance < min_importance_pct% of total.
       These contribute negligibly and can be removed to reduce noise.

    B) REDUNDANT — features in high-correlation pairs (r > threshold).
       For each pair, we recommend removing the LESS important one.
       This catches algebraic duplicates (r=1.000) and near-duplicates (r > 0.98).
       These are reported regardless of importance level.
    """
    total_imp = sum(imp for _, imp in importance)
    threshold = total_imp * min_importance_pct / 100.0

    # Build importance lookup (includes features with 0 importance)
    imp_map: dict[str, float] = {name: imp for name, imp in importance}
    # Also add features that might be in correlation pairs but have 0 importance
    for pair in correlation_pairs:
        for f in [pair["feature_a"], pair["feature_b"]]:
            if f not in imp_map:
                imp_map[f] = 0.0

    # ── Category A: Low-importance features ──
    low_imp = [(name, imp) for name, imp in importance if imp < threshold]

    # ── Category B: Redundant pairs (regardless of importance) ──
    redundant_recommendations = []
    # Track which features we've already decided to remove/keep
    removed_in_redundant: set[str] = set()
    kept_in_redundant: set[str] = set()

    for pair in correlation_pairs:
        imp_a = imp_map.get(pair["feature_a"], 0)
        imp_b = imp_map.get(pair["feature_b"], 0)

        # Determine which to remove (the less important one)
        if imp_a <= imp_b:
            remove_name = pair["feature_a"]
            keep_name = pair["feature_b"]
        else:
            remove_name = pair["feature_b"]
            keep_name = pair["feature_a"]

        # Avoid conflicting recommendations
        if remove_name in kept_in_redundant:
            # The "remove" candidate is already marked as "keep" from another pair
            # Swap: remove the other one instead
            remove_name, keep_name = keep_name, remove_name

        # Determine severity level
        r = abs(pair["pearson_r"])
        if r >= 0.999:
            severity = "critical"    # effectively identical (e.g. algebraic duplicates)
        elif r >= 0.98:
            severity = "high"        # near-duplicate
        elif r >= 0.95:
            severity = "medium"      # strongly redundant
        else:
            severity = "low"         # correlated but possibly distinct

        redundant_recommendations.append({
            "remove": remove_name,
            "keep": keep_name,
            "remove_importance_pct": round(imp_map[remove_name] / total_imp * 100, 4),
            "keep_importance_pct": round(imp_map[keep_name] / total_imp * 100, 4),
            "pearson_r": pair["pearson_r"],
            "severity": severity,
            "reason": (
                f"Redundant with {keep_name} (r={pair['pearson_r']:.4f}); "
                f"keep has higher importance ({imp_map[keep_name]/total_imp*100:.2f}% vs {imp_map[remove_name]/total_imp*100:.2f}%)"
            ),
        })
        removed_in_redundant.add(remove_name)
        kept_in_redundant.add(keep_name)

    # Sort redundant recommendations: critical first, then by importance of the removed feature (lower = safer to remove)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    redundant_recommendations.sort(key=lambda r: (severity_order.get(r["severity"], 9), r["remove_importance_pct"]))

    # ── Combine: Category A (low importance) features ──
    low_imp_prune = []
    for name, imp in low_imp:
        reason = f"Low importance ({imp/total_imp*100:.3f}% of total)"
        # Check if also redundant
        if name in removed_in_redundant:
            r = next(r for r in redundant_recommendations if r["remove"] == name)
            reason += f"; also redundant with {r['keep']} (r={r['pearson_r']:.4f})"
        low_imp_prune.append({
            "feature": name,
            "importance_pct": round(imp / total_imp * 100, 3),
            "reason": reason,
            "category": "low_importance",
        })

    # ── Category B: High-confidence redundant pairs ──
    # Only include critical + high severity (r > 0.98) — these are very safe to act on
    redundant_prune = [
        {
            "feature": r["remove"],
            "importance_pct": r["remove_importance_pct"],
            "reason": r["reason"],
            "category": f"redundant_{r['severity']}",
            "keep_feature": r["keep"],
            "pearson_r": r["pearson_r"],
        }
        for r in redundant_recommendations
        if r["severity"] in ("critical", "high")  # r > 0.98
    ]

    return {
        "low_importance_threshold_pct": min_importance_pct,
        "low_importance_count": len(low_imp),
        "redundant_pairs_count": len(correlation_pairs),
        # Category A: low importance
        "safe_to_prune_low_importance": low_imp_prune[:30],
        # Category B: redundant (critical + high severity)
        "safe_to_prune_redundant": redundant_prune[:30],
        # Combined (for backward compatibility)
        "safe_to_prune": low_imp_prune[:30] + redundant_prune[:30],
        "total_prune_candidates": len(low_imp_prune) + len(redundant_prune),
        "all_redundant_pairs": redundant_recommendations,  # full list for inspection
    }


def analyze_feature_groups(importance: list[tuple[str, float]]) -> dict[str, float]:
    """Aggregate importance by feature group."""
    groups: dict[str, float] = {}
    for name, imp in importance:
        if name.startswith("home_"):
            group = "home_team"
        elif name.startswith("away_"):
            group = "away_team"
        elif name in {"season_series_games", "home_season_series_wins"}:
            group = "h2h"
        elif name in {"same_conference", "same_division"}:
            group = "context_conference"
        elif name in {"home_rest_days", "away_rest_days", "rest_days_diff",
                      "home_b2b", "away_b2b"}:
            group = "context_rest"
        elif name in {"pace_diff", "expected_pace", "home_off_vs_away_def",
                      "away_off_vs_home_def", "net_rtg_diff", "off_def_composite",
                      "total_rating_diff", "efg_diff", "ts_diff", "tov_pct_diff",
                      "ftr_diff", "expected_total"}:
            group = "matchup"
        else:
            group = "other"
        groups[group] = groups.get(group, 0) + imp

    total = sum(groups.values())
    return {k: round(v / total * 100, 1) for k, v in sorted(groups.items(), key=lambda x: -x[1])}


def main(
    version: str | None = None,
    top_n: int = 30,
    correlation_threshold: float = 0.92,
    min_importance_pct: float = 0.1,
    output_json: bool = True,
) -> None:
    """Run feature importance analysis.

    Args:
        version: Model version to analyze (latest if None).
        top_n: Show top N features.
        correlation_threshold: Pearson r threshold for redundancy detection.
        min_importance_pct: Minimum importance % to keep a feature.
        output_json: Save results to JSON file alongside the model.
    """
    print("=" * 70)
    print("NBA Predictor — Feature Importance Analysis")
    print("=" * 70)

    # 1. Load model
    print("\n[1/4] Loading XGBoost model...")
    xgb = XGBoostTrainer()
    xgb_version = xgb.load(version)
    print(f"  Model version: {xgb_version}")

    # 2. Extract feature importance (from all three models)
    print("\n[2/4] Extracting feature importance...")

    all_importance: dict[str, float] = {}
    for target, model_name in [("spread", "model_spread"), ("total", "model_total"), ("win", "model_win")]:
        model = getattr(xgb, model_name, None)
        if model is None:
            print(f"  ⚠ {target} model not loaded, skipping")
            continue

        # Get importance scores
        imp_type = "gain" if hasattr(model, "get_booster") else "weight"
        scores = model.get_booster().get_score(importance_type="gain")

        # Map f-score keys to feature names
        for key, value in scores.items():
            idx = int(key.replace("f", ""))
            if idx < len(FEATURE_NAMES):
                name = FEATURE_NAMES[idx]
                all_importance[name] = all_importance.get(name, 0) + value

    # Normalize to percentages
    total = sum(all_importance.values())
    importance_ranked = sorted(all_importance.items(), key=lambda x: -x[1])
    importance_pct = [(name, round(imp / total * 100, 4)) for name, imp in importance_ranked]

    print(f"  Total features with importance > 0: {len(importance_pct)}")
    print(f"  Top 5: {', '.join(f'{n}({p:.1f}%)' for n, p in importance_pct[:5])}")

    # 3. Print ranking
    print(f"\n[3/4] Feature Importance Ranking (Top {top_n}):")
    print(f"  {'Rank':<5} {'Feature':<40} {'Importance %':>13}")
    print(f"  {'-'*5} {'-'*40} {'-'*13}")
    for rank, (name, imp_pct) in enumerate(importance_pct[:top_n], 1):
        bar = "█" * int(imp_pct * 5)  # scale for visibility
        print(f"  {rank:<5} {name:<40} {imp_pct:>8.2f}%  {bar}")

    # Feature group analysis
    group_imp = analyze_feature_groups(importance_ranked)
    print(f"\n  Importance by Feature Group:")
    for group, imp in group_imp.items():
        print(f"    {group:<20s}: {imp:>6.1f}%")

    # 4. Prune candidates
    print(f"\n[4/4] Prune Analysis:")
    print(f"  Loading dataset for correlation analysis...")

    # Load training data (uses same async pattern as trainer)
    import asyncio

    async def _load_data():
        from ml.src.features.feature_builder import FeatureBuilder
        builder = FeatureBuilder()
        dataset = await builder.build_training_dataset(
            seasons=None,
            lookback_games=15,
            min_games_before=5,
        )
        return dataset

    try:
        dataset = asyncio.run(_load_data())
        print(f"  Dataset: {len(dataset.X)} samples, {dataset.X.shape[1]} features")

        # Correlation analysis
        corr_pairs = compute_correlation_matrix(
            dataset.X, FEATURE_NAMES, threshold=correlation_threshold
        )
        print(f"  High-correlation pairs (r > {correlation_threshold}): {len(corr_pairs)}")

        # Prune candidates
        prune_result = find_prune_candidates(
            importance_ranked, corr_pairs, min_importance_pct=min_importance_pct
        )

        # ── Print results ──

        # Category B: Redundant features (most actionable first)
        redundant_candidates = prune_result.get("safe_to_prune_redundant", [])
        if redundant_candidates:
            print(f"\n  🔴 REDUNDANT features (r > 0.98, safe to remove):")
            for c in redundant_candidates:
                severity_mark = "💀" if c.get("pearson_r", 0) >= 0.999 else "⚠️"
                print(f"    {severity_mark} ✂ {c['feature']:<35s} "
                      f"({c['importance_pct']:.2f}%) → keep {c['keep_feature']} "
                      f"(r={c['pearson_r']:.4f})")

        # Category A: Low importance
        low_imp_candidates = prune_result.get("safe_to_prune_low_importance", [])
        if low_imp_candidates:
            print(f"\n  🟡 Low-importance features (<{min_importance_pct:.1f}% of total):")
            for c in low_imp_candidates[:15]:
                print(f"    ✂ {c['feature']:<40s} ({c['importance_pct']:.2f}%) — {c['reason']}")

        # Summary
        print(f"\n  📊 Summary:")
        print(f"    Critical+High redundant pairs:  {len(redundant_candidates)} features to consider")
        print(f"    Low importance:                {prune_result['low_importance_count']} features")
        if not redundant_candidates and not low_imp_candidates:
            print(f"    ✅ No clear prune candidates found.")

        # Medium-severity pairs for manual review
        all_redundant = prune_result.get("all_redundant_pairs", [])
        medium_pairs = [r for r in all_redundant if r["severity"] == "medium"]
        if medium_pairs:
            print(f"\n  🟠 Medium-severity redundant pairs (0.95 < r < 0.98) — review manually:")
            for r in medium_pairs[:10]:
                print(f"    {r['remove']:<35s} ↔ {r['keep']:<35s} r={r['pearson_r']:.4f}")

        # Save results
        if output_json:
            model_dir = os.path.join(ml_settings.xgboost_dir, xgb_version)
            output_path = os.path.join(model_dir, "feature_analysis.json")

            analysis = {
                "model_version": xgb_version,
                "feature_count": len(FEATURE_NAMES),
                "features_with_importance": len(importance_pct),
                "top_features": [
                    {"rank": i + 1, "feature": name, "importance_pct": imp}
                    for i, (name, imp) in enumerate(importance_pct[:top_n])
                ],
                "feature_group_importance": group_imp,
                "high_correlation_pairs": corr_pairs,
                "prune_analysis": {
                    "total_candidates": prune_result["total_prune_candidates"],
                    "low_importance_threshold_pct": min_importance_pct,
                    "low_importance_count": prune_result["low_importance_count"],
                    "redundant_pairs_count": prune_result["redundant_pairs_count"],
                    "redundant_candidates": redundant_candidates,
                    "low_importance_candidates": low_imp_candidates,
                    "safe_to_prune": prune_result["safe_to_prune"][:30],
                    "all_redundant_pairs": all_redundant,
                },
            }

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"\n  ✅ Analysis saved to {output_path}")

    except Exception as e:
        print(f"  ⚠ Could not compute correlation/prune analysis: {e}")
        print(f"    (This requires a running database with game data)")

    print("\n" + "=" * 70)
    print("Analysis complete! 🏀")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze feature importance")
    parser.add_argument(
        "--version", type=str, default=None,
        help="Model version to analyze (latest if not specified)",
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of top features to show (default: 30)",
    )
    parser.add_argument(
        "--correlation-threshold", type=float, default=0.92,
        help="Pearson r threshold for redundancy (default: 0.92)",
    )
    parser.add_argument(
        "--min-importance-pct", type=float, default=0.1,
        help="Minimum importance %% to keep a feature (default: 0.1)",
    )
    args = parser.parse_args()

    main(
        version=args.version,
        top_n=args.top,
        correlation_threshold=args.correlation_threshold,
        min_importance_pct=args.min_importance_pct,
    )
