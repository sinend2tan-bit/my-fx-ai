# App / Scanner model-contract audit

Audit date: 2026-11-17  
Scope: static source review of `app.py` and `scanner.py` on `main`.  
Status: source-level findings only; no claim of measured feature-value parity or predictive improvement.

## Executive finding

The App and Scanner are not interchangeable implementations of one model. They use different feature schemas, indicator definitions, label entry references, and validation protocols. Their reported scores must not be compared as if they were measurements of the same target.

Do not unify the implementations by renaming columns or copying formulas. First run a controlled, same-input parity experiment and separately assess each model against its own intended deployment target.

## Verified model contracts

| Contract item | App | Scanner |
|---|---|---|
| Model version | 8 | 21 |
| Feature count | 17 | 16 |
| EMA deviation inputs | `Dev_EMA200`, `Dev_EMA20_200` | `Dev_EMA20`, `Dev_EMA200` |
| `Vol_Ratio` | ATR / rolling mean ATR; included in model input | Volume / rolling mean volume (or neutral 1.0); diagnostic only, excluded from model input |
| RSI | rolling mean gains/losses, period 14, min periods 7 | Wilder-style EWM, alpha 1/14 |
| RSI difference | one-bar difference | three-bar difference |
| ATR | rolling mean true range, period 14, min periods 7 | Wilder-style EWM true range, alpha 1/14 |
| ADX | rolling-smoothed directional movement and DX | Wilder-style EWM smoothing |
| ROC | 10-bar percentage change multiplied by 100 | 10-bar percentage change (fraction) |

These differences are source-level facts. Their effect on out-of-sample accuracy is not quantified by this audit.

## Label and validation contract differences

### App

- First-touch target is anchored to the signal bar close.
- Barrier distance is ATR with a spread floor and an additional spread term.
- Same-bar upper/lower ambiguity is labeled neutral.
- Chronological holdout; an embargo of at least the lookahead horizon is applied before the test segment.
- Validation requires all three true classes in train/test/model and balanced accuracy to clear both an absolute floor and a majority-baseline margin.

### Scanner

- First-touch target is anchored to the next bar open.
- Barrier distance is the signal-bar ATR.
- Same-bar upper/lower ambiguity is labeled neutral.
- Three-fold walk-forward validation with purge based on original source-bar positions.
- Model validation and cost-stressed simulated-PnL stability are separate requirements.
- RandomForest scores are explicitly marked uncalibrated.

## Important interpretation constraints

1. App holdout balanced accuracy and Scanner walk-forward OOF balanced accuracy are not directly comparable until target definitions and evaluation populations are aligned.
2. A difference in feature names does not establish a bug. It may be intentional model-version separation.
3. A higher raw RandomForest score is not a calibrated probability or a win probability.
4. A backtest result is not an actual broker execution result. Spread, slippage, swap, gaps, and fill behavior can differ.
5. No feature, target, model, threshold, gate, or live signal should be changed based solely on this static comparison.

## Follow-up experiments (must be isolated and reproducible)

### A. Feature parity experiment

- Build one deterministic synthetic OHLCV fixture with timezone-aware, strictly ordered timestamps.
- Feed identical rows to both indicator pipelines.
- Compare same-semantic columns only; do not compare columns whose definitions differ.
- Record first valid row, NaN count, finite-value count, min/max, mean/std, correlation, and absolute-difference quantiles.
- Explicitly test flat prices, zero volume, missing bars, duplicate timestamps, and short warm-up history.
- Do not overwrite production feature functions or model artifacts.

### B. Target parity experiment

- Use a deterministic OHLC fixture with controlled next-bar opens and barrier touches.
- Compare signal-close versus next-open anchoring, spread floor/cost, same-bar ambiguity, first-touch precedence, and discontinuous windows.
- Report per-class transitions and exclusions; do not force a common label until the intended execution reference is chosen.

### C. Predictive performance experiment

- Use only existing approved development data initially.
- Keep sealed/final data untouched.
- Report per pair, fold, and direction: support, recall, balanced accuracy, baseline, trade count, gross/net pips, cost assumptions, and drawdown.
- Require enough observations before interpreting a direction-specific result.
- Any threshold/feature/model selection must occur inside development folds, not on the final evaluation segment.

## Source locations reviewed

- App feature schema: `app.py` lines 63-68.
- App indicator and target generation: `app.py` around lines 1280-1390.
- App chronological holdout and completeness gates: `app.py` around lines 1570-1695.
- Scanner feature schema: `scanner.py` lines 99-117.
- Scanner indicator calculations: `scanner.py` around lines 1168-1295.
- Scanner target generation: `scanner.py` around lines 1590-1645.
- Scanner source-position purge: `scanner.py` lines 1847-1868.
- Scanner walk-forward and PnL validation: `scanner.py` around lines 1937-2680.

## Safety boundary

This document records source-level findings and a test plan only. It does not authorize or perform model retraining, production feature/label changes, gate loosening, sealed/final data access, paid data/API adoption, broker integration, automatic orders, or deployment.
