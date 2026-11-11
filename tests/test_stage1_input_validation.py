"""段階1（SYN-FX-016）入力検証・C-1 カウンタ・タイムゾーン安全処理のテスト。

承認済み修正仕様書（2026-11-11）の段階1の実装に対する合成テスト。
実データ・実ネットワークは使わない（yfinance はスタブに差し替え、
ソケット接続はガードで禁止する）。

確認すること:
- Scanner（1時間足のみ）: 処理順序 (0) timestamp 不正 → (1) 欠損・非有限値
  → (2) 重複 timestamp (keep="last") → (3) OHLC 整合性 → (4) 特徴量計算
  による除外。各行は最初に該当した理由で一度だけ計上され、C-1 の2つの
  検算式が成立する。
- OHLC 検証条件: High >= max(Open, Close)、Low <= min(Open, Close)、
  High >= Low、価格が有限値かつ正値。等号（High == max(Open, Close) 等）
  は有効。
- keep="last" で残った行が OHLC 不正な場合、古い重複行へ戻らず
  その timestamp 自体を除外する。
- ラベル判定段階は段階1では現行判定の記録のみ。現行コードに不連続を
  理由とするラベル除外は存在しないため dropped_label_discontinuous は 0。
  dropped_label_other（基準価格・ATR 無効によるスキップ）は、入力検証で
  価格 > 0 が保証され、ATR は EWM で正の値を保つため、通常の合成データ
  では 0 になる（理由を捏造せず、実測どおり 0 を確認する）。
- タイムゾーン: 1h の tz-aware index は UTC へ正規化される。tz-naive は
  UTC と推定せず処理を停止する（Scanner 1h/1d、App 1h/1d）。日足は
  段階0で全ペア tz-naive と確認されたため停止対象。
- 対象外の時間足は変更しない: Scanner 4h / App 4h は従来どおり動作し、
  data_quality は付かない。
- モデル版管理: App MODEL_VERSION 7 / Scanner MODEL_VERSION 20。
  旧版（App version 6）の保存済みモデルは再利用されない。品質ゲートを
  通過しなかった場合は旧版モデルへフォールバックせず、その実行の
  評価モデルが使われ、旧ファイルは上書きされない。新版（7）の保存済み
  モデルは従来どおり再利用される。
"""

import ast
import socket
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import pytest
import yfinance as yf
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

import scanner

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app.py"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """実ネットワーク接続を禁止する（スタブ以外の取得経路を塞ぐ）。"""

    def guard(*args, **kwargs):
        raise AssertionError("real network access attempted during stage1 tests")

    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)


# ---------------------------------------------------------------------------
# 共通フィクスチャ
# ---------------------------------------------------------------------------

def make_frame(n=500, tz="UTC", seed=7, start="2026-01-05 00:00", freq="1h"):
    """検証を通過する通常の合成 OHLCV（毎回同じ種で同じ値）。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz=tz)
    close = 150.0 + np.cumsum(rng.normal(0, 0.15, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.01, 0.08, n)
    low = np.minimum(open_, close) - rng.uniform(0.01, 0.08, n)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def assert_checksums(dq):
    """C-1 の2つの検算式が成立すること。"""
    input_drops = (
        dq["dropped_invalid_timestamp"]
        + dq["dropped_missing_or_nonfinite"]
        + dq["dropped_duplicate_timestamp"]
        + dq["dropped_invalid_ohlc"]
        + dq["dropped_feature_nan"]
    )
    assert dq["feature_rows"] == dq["input_rows"] - input_drops
    label_drops = (
        dq["dropped_label_discontinuous"]
        + dq["dropped_label_insufficient_future"]
        + dq["dropped_label_other"]
    )
    assert dq["trainable_rows"] == dq["feature_rows"] - label_drops


def stub_scanner_download(monkeypatch, frame):
    monkeypatch.setattr(yf, "download", lambda *a, **k: frame.copy())


# ---------------------------------------------------------------------------
# Scanner: 入力検証と C-1 カウンタ
# ---------------------------------------------------------------------------

def test_scanner_clean_frame_counters(monkeypatch):
    stub_scanner_download(monkeypatch, make_frame())
    out = scanner.bars("USDJPY=X", interval="1h", period="180d")
    assert out is not None
    dq = out.attrs.get("data_quality")
    assert dq is not None
    # 末尾の未確定足1本を除いた 499 件が検証パイプラインの入力。
    assert dq["input_rows"] == 499
    assert dq["dropped_invalid_timestamp"] == 0
    assert dq["dropped_missing_or_nonfinite"] == 0
    assert dq["dropped_duplicate_timestamp"] == 0
    assert dq["dropped_invalid_ohlc"] == 0
    # ラベル段階: 末尾8本が未来窓不足、不連続・その他は現行判定では 0。
    assert dq["dropped_label_discontinuous"] == 0
    assert dq["dropped_label_insufficient_future"] == 8
    assert dq["dropped_label_other"] == 0
    assert dq["trainable_rows"] == len(out)
    assert_checksums(dq)


def test_scanner_invalid_ohlc_variants(monkeypatch):
    frame = make_frame()
    hi, lo, cl = (frame.columns.get_loc(c) for c in ("High", "Low", "Close"))
    # High < Low
    frame.iloc[100, hi] = frame.iloc[100]["Low"] - 0.5
    # Low > min(Open, Close)（Low > Close にする）
    frame.iloc[150, lo] = frame.iloc[150]["Close"] + 0.5
    # High < max(Open, Close)（High < Open にする）
    frame.iloc[200, hi] = frame.iloc[200]["Open"] - 0.5
    # 価格が正値でない（Close = 0）
    frame.iloc[250, cl] = 0.0
    # 等号ケース（有効でなければならない）: High == max(Open, Close) かつ
    # Low == min(Open, Close)（範囲は非ゼロにして特徴量計算は通過させる）。
    doji_ts = frame.index[300]
    close_v = float(frame.iloc[300]["Close"])
    open_v = close_v + 0.10
    frame.iloc[300, :] = [open_v, open_v, close_v, close_v, 1_000_000.0]
    stub_scanner_download(monkeypatch, frame)
    out = scanner.bars("USDJPY=X", interval="1h", period="180d")
    assert out is not None
    dq = out.attrs["data_quality"]
    assert dq["dropped_invalid_ohlc"] == 4
    assert dq["dropped_missing_or_nonfinite"] == 0
    assert_checksums(dq)
    # 等号の行は除外されず、特徴量フレームに残る。
    assert doji_ts in out.attrs["latest_features"].index


def test_scanner_missing_and_nonfinite(monkeypatch):
    frame = make_frame()
    cl, hi, lo = (frame.columns.get_loc(c) for c in ("Close", "High", "Low"))
    frame.iloc[100, cl] = np.nan
    frame.iloc[150, hi] = np.inf
    frame.iloc[200, lo] = -np.inf
    stub_scanner_download(monkeypatch, frame)
    out = scanner.bars("USDJPY=X", interval="1h", period="180d")
    assert out is not None
    dq = out.attrs["data_quality"]
    assert dq["dropped_missing_or_nonfinite"] == 3
    assert dq["dropped_invalid_ohlc"] == 0
    assert_checksums(dq)


def test_scanner_duplicate_keep_last(monkeypatch):
    frame = make_frame()
    ts = frame.index[350]
    dup = frame.iloc[[350]].copy()
    dup.iloc[0, dup.columns.get_loc("Close")] = dup.iloc[0]["Close"] + 0.03
    # High/Low の範囲内に収まるよう調整（OHLC 整合性を保つ）。
    dup.iloc[0, dup.columns.get_loc("High")] = max(
        dup.iloc[0]["High"], dup.iloc[0]["Close"]
    )
    combined = pd.concat([frame.iloc[:351], dup, frame.iloc[351:]])
    stub_scanner_download(monkeypatch, combined)
    out = scanner.bars("USDJPY=X", interval="1h", period="180d")
    assert out is not None
    dq = out.attrs["data_quality"]
    assert dq["input_rows"] == 500  # 501 本ダウンロード − 未確定足 1 本
    assert dq["dropped_duplicate_timestamp"] == 1
    assert_checksums(dq)
    # keep="last": 後ろの複製行の値が残る。
    assert out.attrs["latest_features"].loc[ts, "Close"] == dup.iloc[0]["Close"]


def test_scanner_duplicate_last_copy_invalid_drops_timestamp(monkeypatch):
    frame = make_frame()
    ts = frame.index[350]
    dup = frame.iloc[[350]].copy()
    # 後ろの複製行だけを OHLC 不正（High < Low）にする。
    dup.iloc[0, dup.columns.get_loc("High")] = dup.iloc[0]["Low"] - 0.5
    combined = pd.concat([frame.iloc[:351], dup, frame.iloc[351:]])
    stub_scanner_download(monkeypatch, combined)
    out = scanner.bars("USDJPY=X", interval="1h", period="180d")
    assert out is not None
    dq = out.attrs["data_quality"]
    # 先の複製は重複として、残った不正行は OHLC 違反として計上され、
    # その timestamp 自体が除外される（古い複製行へは戻らない）。
    assert dq["dropped_duplicate_timestamp"] == 1
    assert dq["dropped_invalid_ohlc"] == 1
    assert ts not in out.attrs["latest_features"].index
    assert_checksums(dq)


def test_scanner_invalid_timestamp_nat(monkeypatch):
    frame = make_frame()
    nat_row = frame.iloc[[10]].copy()
    nat_row.index = pd.DatetimeIndex([pd.NaT])
    combined = pd.concat([frame, nat_row])
    stub_scanner_download(monkeypatch, combined)
    out = scanner.bars("USDJPY=X", interval="1h", period="180d")
    assert out is not None
    dq = out.attrs["data_quality"]
    assert dq["dropped_invalid_timestamp"] == 1
    assert_checksums(dq)


def test_scanner_non_datetime_index_stops(monkeypatch):
    frame = make_frame()
    frame.index = pd.RangeIndex(len(frame))
    stub_scanner_download(monkeypatch, frame)
    assert scanner.bars("USDJPY=X", interval="1h", period="180d") is None


# ---------------------------------------------------------------------------
# Scanner: タイムゾーン安全処理と対象外時間足
# ---------------------------------------------------------------------------

def test_scanner_tz_naive_1h_stops(monkeypatch):
    frame = make_frame()
    frame.index = frame.index.tz_localize(None)
    stub_scanner_download(monkeypatch, frame)
    assert scanner.bars("USDJPY=X", interval="1h", period="180d") is None


def test_scanner_tz_naive_daily_stops(monkeypatch):
    # 段階0の実測どおり、日足は tz-naive のため分析・ラベル処理を停止する。
    frame = make_frame()
    frame.index = frame.index.tz_localize(None)
    stub_scanner_download(monkeypatch, frame)
    assert scanner.bars("USDJPY=X", interval="1d", period="2y") is None


def test_scanner_tz_aware_daily_legacy_path(monkeypatch):
    # tz-aware な日足は従来経路のまま処理され、段階1のカウンタは付かない
    # （段階1の対象は1時間足のみ）。
    stub_scanner_download(monkeypatch, make_frame())
    out = scanner.bars("USDJPY=X", interval="1d", period="2y")
    assert out is not None
    assert "data_quality" not in out.attrs


def test_scanner_4h_not_stopped_by_stage1_gate(monkeypatch):
    # 4時間足は段階1の対象外。tz-naive な1h入力でも段階1のゲートでは
    # 停止せず、従来の4h処理（段階2の対象）が維持される。
    frame = make_frame(n=2000, seed=11)
    frame.index = frame.index.tz_localize(None)
    stub_scanner_download(monkeypatch, frame)
    out = scanner.bars("USDJPY=X", interval="4h", period="90d")
    assert out is not None
    assert "data_quality" not in out.attrs


def test_scanner_version_and_status_record():
    assert scanner.MODEL_VERSION == 20
    source = (REPO_ROOT / "scanner.py").read_text(encoding="utf-8")
    assert "v19_" not in source
    dq = {
        "input_rows": 100,
        "dropped_invalid_timestamp": 0,
        "dropped_missing_or_nonfinite": 1,
        "dropped_duplicate_timestamp": 0,
        "dropped_invalid_ohlc": 2,
        "dropped_feature_nan": 3,
        "feature_rows": 94,
        "dropped_label_discontinuous": 0,
        "dropped_label_insufficient_future": 8,
        "dropped_label_other": 0,
        "trainable_rows": 86,
    }
    record = scanner.build_status_record(
        "2026-11-12T00:00:00+09:00",
        "2026-11-12T00:05:00+09:00",
        [{"name": "米ドル/円 (USD/JPY)", "ticker": "USDJPY=X", "data_quality": dq}],
        {},
        {},
    )
    assert record["pairs"][0]["data_quality"] == dq


# ---------------------------------------------------------------------------
# App: AST 抽出による load_and_process_data の検証（PR #25 と同じ手法）
# ---------------------------------------------------------------------------

_APP_SRC = APP_PATH.read_text(encoding="utf-8")
_APP_TREE = ast.parse(_APP_SRC)


def _extract_function(name):
    for node in _APP_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            fn = ast.parse(ast.unparse(node)).body[0]
            fn.decorator_list = []
            return fn
    raise KeyError(name)


def _app_literal(name):
    for node in _APP_TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class _StStub:
    def __init__(self):
        self.warnings = []
        self.session_state = {}

    def warning(self, message):
        self.warnings.append(str(message))


def _app_namespace(frame_factory):
    st = _StStub()

    class _YF:
        @staticmethod
        def download(*args, **kwargs):
            return frame_factory()

    ns = {
        "pd": pd,
        "np": np,
        "yf": _YF,
        "st": st,
        "FEATURE_COLUMNS": _app_literal("FEATURE_COLUMNS"),
        "DEFAULT_SPREAD_PIPS": _app_literal("DEFAULT_SPREAD_PIPS"),
        "LOOKAHEAD_BARS": _app_literal("LOOKAHEAD_BARS"),
    }
    for fname in ("flatten_yf_df", "load_and_process_data"):
        code = compile(
            ast.Module(body=[_extract_function(fname)], type_ignores=[]),
            "app_extract",
            "exec",
        )
        exec(code, ns)
    return ns, st


def test_app_clean_frame_counters():
    base = make_frame()
    ns, st = _app_namespace(lambda: base.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "60d", "1h")
    assert out is not None
    assert st.warnings == []
    dq = out.attrs.get("data_quality")
    assert dq is not None
    assert dq["input_rows"] == 499
    assert dq["dropped_invalid_timestamp"] == 0
    assert dq["dropped_missing_or_nonfinite"] == 0
    assert dq["dropped_duplicate_timestamp"] == 0
    assert dq["dropped_invalid_ohlc"] == 0
    assert dq["dropped_label_discontinuous"] == 0
    assert dq["dropped_label_insufficient_future"] == 8
    assert dq["dropped_label_other"] == 0
    # App の学習対象行数 = 特徴量行 − ラベルなし（末尾の未来窓不足）行。
    assert dq["trainable_rows"] == dq["feature_rows"] - 8
    assert int(out["Target"].notna().sum()) == dq["trainable_rows"]
    assert_checksums(dq)


def test_app_dirty_frame_counters():
    frame = make_frame()
    hi, lo, cl = (frame.columns.get_loc(c) for c in ("High", "Low", "Close"))
    frame.iloc[100, hi] = frame.iloc[100]["Low"] - 0.5  # High < Low
    frame.iloc[150, cl] = 0.0  # 正値でない
    frame.iloc[200, cl] = np.nan  # 欠損
    frame.iloc[250, hi] = np.inf  # 非有限値
    # 重複 timestamp（後ろが有効）: keep="last" で残る。
    ts = frame.index[350]
    dup = frame.iloc[[350]].copy()
    new_close = float(dup.iloc[0]["Close"] + 0.03)
    dup.iloc[0, cl] = new_close
    dup.iloc[0, hi] = max(float(dup.iloc[0]["High"]), new_close)
    combined = pd.concat([frame.iloc[:351], dup, frame.iloc[351:]])
    # NaT 行（timestamp 不正）。
    nat_row = frame.iloc[[10]].copy()
    nat_row.index = pd.DatetimeIndex([pd.NaT])
    combined = pd.concat([combined, nat_row])
    ns, _st = _app_namespace(lambda: combined.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "60d", "1h")
    assert out is not None
    dq = out.attrs["data_quality"]
    assert dq["dropped_invalid_timestamp"] == 1
    assert dq["dropped_missing_or_nonfinite"] == 2
    assert dq["dropped_duplicate_timestamp"] == 1
    assert dq["dropped_invalid_ohlc"] == 2
    assert out.loc[ts, "Close"] == new_close
    assert_checksums(dq)


def test_app_duplicate_last_copy_invalid_drops_timestamp():
    frame = make_frame()
    ts = frame.index[350]
    dup = frame.iloc[[350]].copy()
    dup.iloc[0, dup.columns.get_loc("High")] = dup.iloc[0]["Low"] - 0.5
    combined = pd.concat([frame.iloc[:351], dup, frame.iloc[351:]])
    ns, _st = _app_namespace(lambda: combined.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "60d", "1h")
    assert out is not None
    dq = out.attrs["data_quality"]
    assert dq["dropped_duplicate_timestamp"] == 1
    assert dq["dropped_invalid_ohlc"] == 1
    assert ts not in out.index
    assert_checksums(dq)


def test_app_tz_aware_converted_to_utc():
    frame = make_frame(tz="Asia/Tokyo")
    ns, _st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "60d", "1h")
    assert out is not None
    assert str(out.index.tz) == "UTC"


def test_app_tz_naive_1h_stops_with_message():
    frame = make_frame()
    frame.index = frame.index.tz_localize(None)
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "60d", "1h")
    assert out is None
    assert any("タイムゾーン" in w and "停止" in w for w in st.warnings)


def test_app_tz_naive_daily_stops_with_message():
    frame = make_frame()
    frame.index = frame.index.tz_localize(None)
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "2y", "1d")
    assert out is None
    assert any("タイムゾーン" in w and "停止" in w for w in st.warnings)


def test_app_4h_legacy_path_unchanged():
    # 4時間足は段階1の対象外: tz-naive な1h入力でも段階1のゲートでは
    # 停止せず、従来のリサンプル処理が維持され、カウンタは付かない。
    frame = make_frame(n=2000, seed=11)
    frame.index = frame.index.tz_localize(None)
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert out is not None
    assert "data_quality" not in out.attrs
    assert st.warnings == []


# ---------------------------------------------------------------------------
# App: モデル版管理（旧モデルの再利用防止・非フォールバック・新版の再利用）
# ---------------------------------------------------------------------------

def _model_namespace():
    import os
    import time
    from datetime import datetime

    st = _StStub()
    ns = {
        "pd": pd,
        "np": np,
        "st": st,
        "os": os,
        "time": time,
        "joblib": joblib,
        "RandomForestClassifier": RandomForestClassifier,
        "DummyClassifier": DummyClassifier,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "compute_sample_weight": compute_sample_weight,
        "FEATURE_COLUMNS": _app_literal("FEATURE_COLUMNS"),
        "LOOKAHEAD_BARS": _app_literal("LOOKAHEAD_BARS"),
        "MODEL_VERSION": _app_literal("MODEL_VERSION"),
        "datetime": datetime,
        "JST": ZoneInfo("Asia/Tokyo"),
    }
    for fname in ("_model_filename", "analyze_signal_with_backtest"):
        code = compile(
            ast.Module(body=[_extract_function(fname)], type_ignores=[]),
            "app_extract",
            "exec",
        )
        exec(code, ns)
    return ns, st


def _model_df(mode, n=500, seed=3):
    rng = np.random.default_rng(seed)
    columns = _app_literal("FEATURE_COLUMNS")
    idx = pd.date_range("2026-01-05", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame(rng.normal(0, 1, (n, len(columns))), columns=columns, index=idx)
    if mode == "learnable":
        r1 = df["Return_1"].to_numpy()
        y = np.where(r1 > 0.2, 1, np.where(r1 < -0.2, -1, 0))
    else:  # noise: 特徴量と無関係なラベル（品質ゲートを通過しない）
        y = rng.choice([-1, 0, 1], size=n)
    df["Target"] = y.astype(float)
    df["Close"] = 150.0 + np.cumsum(rng.normal(0, 0.1, n))
    df["RSI"] = 50.0
    df["ADX"] = 30.0
    df["ATR_Ratio"] = 0.005
    return df


def _tiny_model(n_estimators=10, single_class=False, seed=3):
    rng = np.random.default_rng(seed)
    columns = _app_literal("FEATURE_COLUMNS")
    X = rng.normal(0, 1, (60, len(columns)))
    if single_class:
        y = np.ones(60, dtype=int)
    else:
        y = rng.choice([-1, 0, 1], size=60)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    model.fit(X, y)
    return model


def _save_model_file(ns, tmp_path, monkeypatch, version, model):
    monkeypatch.chdir(tmp_path)
    filename = ns["_model_filename"]("USDJPY=X", "1h")
    joblib.dump(
        {
            "version": version,
            "features": _app_literal("FEATURE_COLUMNS"),
            "model": model,
            "trained_at": "2026-11-12T00:00:00+09:00",
            "feedback_applied": 0,
        },
        filename,
    )
    return tmp_path / filename


def test_app_model_version_is_7():
    assert _app_literal("MODEL_VERSION") == 7


def test_app_old_version_model_not_reused(tmp_path, monkeypatch):
    ns, st = _model_namespace()
    path = _save_model_file(ns, tmp_path, monkeypatch, 6, _tiny_model())
    result = ns["analyze_signal_with_backtest"](
        _model_df("learnable"), None, "USDJPY=X", "1h"
    )
    assert result is not None
    # 学習可能なデータでは品質ゲートを通過し、新版（7）で保存し直される。
    # 旧版ファイルが再利用されていれば、この保存は起きない。
    assert st.session_state["model_eval_metrics"]["quality_ok"] is True
    saved = joblib.load(path)
    assert saved["version"] == 7
    assert saved["model"].n_estimators == 260  # 新規学習した候補モデル


def test_app_gate_failure_no_fallback_to_old_model(tmp_path, monkeypatch):
    ns, st = _model_namespace()
    # 旧版モデルは単一クラスだけを返す物にする。再利用（フォールバック）
    # されれば特徴量重要度が全て 0 になるため、混入を検出できる。
    path = _save_model_file(
        ns, tmp_path, monkeypatch, 6, _tiny_model(single_class=True)
    )
    before = path.read_bytes()
    result = ns["analyze_signal_with_backtest"](
        _model_df("noise"), None, "USDJPY=X", "1h"
    )
    assert result is not None
    # 品質ゲート不通過: 旧ファイルは上書きされず、使われるのは当該実行の
    # 評価モデル（重要度の合計が正）であり、旧版モデルではない。
    assert st.session_state["model_eval_metrics"]["quality_ok"] is False
    assert path.read_bytes() == before
    importances = result[6]
    assert float(importances.sum()) > 0


def test_app_current_version_model_reused(tmp_path, monkeypatch):
    ns, _st = _model_namespace()
    path = _save_model_file(ns, tmp_path, monkeypatch, 7, _tiny_model())
    before = path.read_bytes()
    result = ns["analyze_signal_with_backtest"](
        _model_df("learnable"), None, "USDJPY=X", "1h"
    )
    assert result is not None
    # 新版（7）で鮮度内の保存済みモデルは従来どおり再利用され、
    # ファイルは書き換わらない（再利用の仕組み自体は壊していない）。
    assert path.read_bytes() == before
