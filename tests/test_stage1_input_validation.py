"""段階1（SYN-FX-016）入力検証・C-1 カウンタ・タイムゾーン安全処理のテスト。

承認済み修正仕様書（2026-11-11）の段階1の実装に対する合成テスト。
実データ・実ネットワークは使わない（yfinance はスタブに差し替え、
ソケット接続はガードで禁止する）。

確認すること:
- Scanner（1時間足のみ）: 処理順序は、DatetimeIndex 確認 → タイム
  ゾーン確認・UTC 正規化 → 最新 timestamp 群の鮮度除外 → (0) timestamp
  不正 → (1) 欠損・非有限値 → (2) 重複 timestamp (keep="last")
  → (3) OHLC 整合性 → (4) 特徴量計算による除外。各行は最初に該当した
  理由で一度だけ計上され、C-1 の2つの検算式が成立する。
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
- 対象外の時間足は変更しない: Scanner 4h は従来どおり動作する。
  App 4h は段階1では従来どおりだったが、段階2（SYN-FX-017）で UTC
  統一・完全性判定に変更された（tests/test_stage2_4h_aggregation.py）。
  どちらの段階でも 4h に data_quality は付かない。
- モデル版管理: App MODEL_VERSION 8（段階1で 6→7、段階2で 7→8）/
  Scanner MODEL_VERSION 21。
  旧版（App version 7）の保存済みモデルは再利用されない。品質ゲートを
  通過しなかった場合は旧版モデルへフォールバックせず、その実行の
  評価モデルが使われ、旧ファイルは上書きされない。新版（8）の保存済み
  モデルは従来どおり再利用される。
- C-1 修正（案A、ChatGPT 決定）: 鮮度除外を品質検証の前に置く。
  DatetimeIndex 確認 → タイムゾーン確認（tz-aware は UTC 正規化、
  tz-naive は停止）→ 最新 timestamp 群の鮮度除外 → input_rows 確定
  → (0) timestamp 不正 → (1) 欠損・非有限値 → (2) 重複 timestamp
  → (3) OHLC 整合性 → (4) 特徴量計算による除外。最終取得行の各
  ケース（正常 / NaN / +Inf / -Inf / OHLC不正 / 最新 timestamp の
  複数行 / NaT / 最新より前の重複）で、検算式が意味的にも成立し、
  学習対象行数が実際の処理結果と一致すること、最新足の品質不正が
  有効な直前確定足を追加で除外しないことを確認する。
- 追加レビュー対応（項目1）: App の停止理由はキャッシュ対象のコアが
  戻り値で返し、公開ラッパーがキャッシュの外側で記録・表示する。
  停止理由の記録・区別・解除をここで確認し、キャッシュミス／ヒット
  時の表示は tests/test_stage1_app_cache.py（AppTest）で確認する。
- 再レビュー対応（適用範囲の是正）: 段階1の品質検証・C-1 カウンタは
  App の1時間足だけに限定する。日足は、DatetimeIndex でない場合と
  tz-naive の場合の安全停止だけを維持し、tz-aware の場合は従来
  （base）経路で処理されること（重複・OHLC 不正行が残り、
  data_quality が付かないこと、段階1で無変更の従来経路と出力が
  一致すること）を確認する。
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
    assert scanner.MODEL_VERSION == 21
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
        # 停止理由の記録はテストごとに新しい空のレジストリを使う。
        "_LOAD_STOP_REASONS": {},
        "_LOAD_STOP_WARNED": set(),
    }
    for fname in (
        "flatten_yf_df",
        "_load_and_process_data_cached",
        "get_load_stop_reason",
        "load_and_process_data",
    ):
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


def test_app_4h_tz_naive_stops_after_stage2():
    # 段階1では4時間足は対象外で、tz-naive な1h入力でも停止せず従来の
    # リサンプル処理が維持されていた。段階2（SYN-FX-017）で4時間足にも
    # 実行時タイムゾーン検査が適用され、tz-naive は UTC と推定せず
    # 停止するようになった（新しい期待値は段階2の実装 PR で更新。
    # 4h の集約そのものの検証は tests/test_stage2_4h_aggregation.py）。
    frame = make_frame(n=2000, seed=11)
    frame.index = frame.index.tz_localize(None)
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert out is None
    assert any("タイムゾーン" in w and "停止" in w for w in st.warnings)
    reason = ns["get_load_stop_reason"]("USDJPY=X", "4h")
    assert reason is not None
    assert reason["code"] == "tz_unavailable"


def test_app_daily_legacy_path_keeps_dirty_rows():
    # 日足は段階1の適用範囲外（再レビューで是正）: tz-aware の日足は
    # 従来（base）経路で処理され、重複 timestamp の整理・OHLC 整合性
    # 検証・カウンタ計上は行われない。欠損行だけは従来の dropna で
    # 落ち、末尾の未確定足は従来どおり iloc[:-1] で落ちる。
    frame = make_frame(n=500, tz="UTC", start="2024-06-03", freq="1d")
    cl, hi = (frame.columns.get_loc(c) for c in ("Close", "High"))
    nan_ts = frame.index[100]
    frame.iloc[100, cl] = np.nan  # 欠損行
    ohlc_ts = frame.index[200]
    frame.iloc[200, hi] = frame.iloc[200]["Low"] - 0.5  # OHLC 不正（High < Low）
    dup_ts = frame.index[350]
    dup = frame.iloc[[350]].copy()
    new_close = float(dup.iloc[0]["Close"]) + 0.03
    dup.iloc[0, cl] = new_close
    dup.iloc[0, hi] = max(float(dup.iloc[0]["High"]), new_close)
    combined = pd.concat([frame.iloc[:351], dup, frame.iloc[351:]])
    ns, st = _app_namespace(lambda: combined.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "2y", "1d")
    assert out is not None
    assert st.warnings == []
    # 1時間足用の data_quality は日足に付与されない。
    assert "data_quality" not in out.attrs
    # 従来経路の帰結: 欠損行は落ち、OHLC 不正行と重複行は残る。
    assert nan_ts not in out.index
    assert ohlc_ts in out.index
    assert int((out.index == dup_ts).sum()) == 2
    # 末尾の未確定足は従来どおり落ちる。
    assert combined.index[-1] not in out.index


def test_app_daily_output_matches_legacy_interval_path():
    # 日足の処理が、段階1で無変更の従来経路（15分足が通る else 経路＝
    # base の処理）と完全一致することで、従来動作の維持を示す。
    frame = make_frame(n=500, tz="UTC", start="2024-06-03", freq="1d")
    ns, _st = _app_namespace(lambda: frame.copy())
    out_daily = ns["load_and_process_data"]("USDJPY=X", "2y", "1d")
    out_legacy = ns["load_and_process_data"]("USDJPY=X", "2y", "15m")
    assert out_daily is not None and out_legacy is not None
    pd.testing.assert_frame_equal(out_daily, out_legacy)


def test_app_daily_tz_aware_nonutc_processed_without_conversion():
    # tz-aware の日足は tz 変換も含めて従来経路で処理する（1時間足の
    # UTC 正規化は日足には適用しない）。index のタイムゾーンは
    # そのまま保持される。
    frame = make_frame(n=500, tz="Asia/Tokyo", start="2024-06-03", freq="1d")
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "2y", "1d")
    assert out is not None
    assert st.warnings == []
    assert str(out.index.tz) == "Asia/Tokyo"
    assert "data_quality" not in out.attrs


def test_app_daily_non_datetime_index_stops():
    # 日足でも、DatetimeIndex でない場合はタイムゾーン安全仕様として
    # 停止し、理由コードが記録される。
    frame = make_frame(n=500, tz="UTC", start="2024-06-03", freq="1d")
    frame.index = pd.RangeIndex(len(frame))
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "2y", "1d")
    assert out is None
    reason = ns["get_load_stop_reason"]("USDJPY=X", "1d")
    assert reason is not None
    assert reason["code"] == "timestamp_format_unconfirmed"
    assert any("時刻形式" in w and "停止" in w for w in st.warnings)


# ---------------------------------------------------------------------------
# App: モデル版管理（旧モデルの再利用防止・非フォールバック・新版の再利用）
# ---------------------------------------------------------------------------

def _model_namespace():
    import os
    import tempfile
    import time
    from datetime import datetime

    st = _StStub()
    ns = {
        "pd": pd,
        "np": np,
        "st": st,
        "os": os,
        "tempfile": tempfile,
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
    # AIFX-APP-PERSIST-20261113-01: 保存処理が _save_trained_model に
    # 分離されたため、抽出対象へ追加する（依存の列挙は従来どおり）。
    for fname in ("_model_filename", "_save_trained_model", "analyze_signal_with_backtest"):
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


def test_app_model_version_is_8():
    # 段階1で 6→7、段階2（SYN-FX-017）で 7→8。
    assert _app_literal("MODEL_VERSION") == 8


def test_app_old_version_model_not_reused(tmp_path, monkeypatch):
    ns, st = _model_namespace()
    path = _save_model_file(ns, tmp_path, monkeypatch, 7, _tiny_model())
    result = ns["analyze_signal_with_backtest"](
        _model_df("learnable"), None, "USDJPY=X", "1h"
    )
    assert result is not None
    # 学習可能なデータでは品質ゲートを通過し、新版（8）で保存し直される。
    # 旧版ファイルが再利用されていれば、この保存は起きない。
    assert st.session_state["model_eval_metrics"]["quality_ok"] is True
    saved = joblib.load(path)
    assert saved["version"] == 8
    assert saved["model"].n_estimators == 260  # 新規学習した候補モデル


def test_app_gate_failure_no_fallback_to_old_model(tmp_path, monkeypatch):
    ns, st = _model_namespace()
    # 旧版モデルは単一クラスだけを返す物にする。再利用（フォールバック）
    # されれば特徴量重要度が全て 0 になるため、混入を検出できる。
    path = _save_model_file(
        ns, tmp_path, monkeypatch, 7, _tiny_model(single_class=True)
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
    path = _save_model_file(ns, tmp_path, monkeypatch, 8, _tiny_model())
    before = path.read_bytes()
    result = ns["analyze_signal_with_backtest"](
        _model_df("learnable"), None, "USDJPY=X", "1h"
    )
    assert result is not None
    # 新版（8）で鮮度内の保存済みモデルは従来どおり再利用され、
    # ファイルは書き換わらない（再利用の仕組み自体は壊していない）。
    assert path.read_bytes() == before


def test_app_forced_retrain_replaces_old_model_only_after_validation(tmp_path, monkeypatch):
    ns, st = _model_namespace()
    path = _save_model_file(ns, tmp_path, monkeypatch, 8, _tiny_model())
    before = path.read_bytes()

    result = ns["analyze_signal_with_backtest"](
        _model_df("learnable"), None, "USDJPY=X", "1h", force_retrain=True
    )

    assert result is not None
    assert st.session_state["model_eval_metrics"]["quality_ok"] is True
    saved = joblib.load(path)
    assert saved["version"] == 8
    assert saved["model"].n_estimators == 260
    assert path.read_bytes() != before
    assert list(tmp_path.glob(".aifx-model-*.tmp")) == []


def test_app_forced_retrain_gate_failure_preserves_old_model(tmp_path, monkeypatch):
    ns, st = _model_namespace()
    path = _save_model_file(ns, tmp_path, monkeypatch, 8, _tiny_model())
    before = path.read_bytes()

    result = ns["analyze_signal_with_backtest"](
        _model_df("noise"), None, "USDJPY=X", "1h", force_retrain=True
    )

    assert result is not None
    assert st.session_state["model_eval_metrics"]["quality_ok"] is False
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".aifx-model-*.tmp")) == []


def test_app_atomic_save_dump_failure_preserves_old_model(tmp_path, monkeypatch):
    ns, _st = _model_namespace()
    path = _save_model_file(ns, tmp_path, monkeypatch, 8, _tiny_model())
    before = path.read_bytes()

    def fail_dump(*args, **kwargs):
        raise OSError("private simulated dump error")

    monkeypatch.setattr(ns["joblib"], "dump", fail_dump)
    payload = {
        "version": 8,
        "features": _app_literal("FEATURE_COLUMNS"),
        "model": _tiny_model(),
    }
    assert ns["_save_trained_model"](str(path), payload) is False
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".aifx-model-*.tmp")) == []


def test_app_atomic_save_reload_failure_preserves_old_model(tmp_path, monkeypatch):
    ns, _st = _model_namespace()
    path = _save_model_file(ns, tmp_path, monkeypatch, 8, _tiny_model())
    before = path.read_bytes()

    def fail_load(*args, **kwargs):
        raise OSError("private simulated reload error")

    monkeypatch.setattr(ns["joblib"], "load", fail_load)
    payload = {
        "version": 8,
        "features": _app_literal("FEATURE_COLUMNS"),
        "model": _tiny_model(),
    }
    assert ns["_save_trained_model"](str(path), payload) is False
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".aifx-model-*.tmp")) == []


@pytest.mark.parametrize("invalid_kind", ["version", "features", "model_type", "single_class"])
def test_app_atomic_save_rejects_invalid_candidate_and_preserves_old_model(
    tmp_path, monkeypatch, invalid_kind
):
    ns, _st = _model_namespace()
    path = _save_model_file(ns, tmp_path, monkeypatch, 8, _tiny_model())
    before = path.read_bytes()
    candidate = _tiny_model(single_class=(invalid_kind == "single_class"))
    payload = {
        "version": 7 if invalid_kind == "version" else 8,
        "features": ["wrong"] if invalid_kind == "features" else _app_literal("FEATURE_COLUMNS"),
        "model": object() if invalid_kind == "model_type" else candidate,
    }
    monkeypatch.setattr(ns["joblib"], "load", lambda _path: payload)

    assert ns["_save_trained_model"](str(path), {
        "version": 8,
        "features": _app_literal("FEATURE_COLUMNS"),
        "model": candidate,
    }) is False
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".aifx-model-*.tmp")) == []


def test_app_atomic_save_replace_failure_preserves_old_model(tmp_path, monkeypatch):
    ns, _st = _model_namespace()
    path = _save_model_file(ns, tmp_path, monkeypatch, 8, _tiny_model())
    before = path.read_bytes()

    def fail_replace(*args, **kwargs):
        raise OSError("private simulated replace error")

    monkeypatch.setattr(ns["os"], "replace", fail_replace)
    payload = {
        "version": 8,
        "features": _app_literal("FEATURE_COLUMNS"),
        "model": _tiny_model(),
    }
    assert ns["_save_trained_model"](str(path), payload) is False
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".aifx-model-*.tmp")) == []


def test_app_atomic_save_without_old_model_leaves_no_partial_file_on_failure(
    tmp_path, monkeypatch
):
    ns, _st = _model_namespace()
    monkeypatch.chdir(tmp_path)

    def fail_replace(*args, **kwargs):
        raise OSError("private simulated replace error")

    monkeypatch.setattr(ns["os"], "replace", fail_replace)
    payload = {
        "version": 8,
        "features": _app_literal("FEATURE_COLUMNS"),
        "model": _tiny_model(),
    }
    assert ns["_save_trained_model"]("model_USDJPY_1h.joblib", payload) is False
    assert not (tmp_path / "model_USDJPY_1h.joblib").exists()
    assert list(tmp_path.glob(".aifx-model-*.tmp")) == []


def test_app_force_retrain_button_does_not_delete_existing_model():
    source = APP_PATH.read_text(encoding="utf-8")
    start = source.index('if st.sidebar.button("🧠 AIモデルを今すぐ強制再学習 (知見反映)")')
    end = source.index("m_file_info =", start)
    button_block = source[start:end]
    assert "os.remove" not in button_block
    assert "force_retrain_once" in button_block
    assert "st.cache_data.clear()" in button_block


# ---------------------------------------------------------------------------
# C-1 修正（案A）: 鮮度除外を品質検証の前に置く処理順序のケース検証
#
# ChatGPT 決定（案A採用、2026-11-12）に基づく実装後の挙動を固定する。
# 旧実装は品質検証 (0)〜(3) をダウンロード全行に対して行い、その後に
# 最新の残存行を iloc[:-1] で鮮度除外していたため、最終行が不正な
# 場合に「不正行が品質カウンタに計上され、鮮度除外が有効な直前確定足
# に移って、その確定足が無記録で失われる」という帰属の入れ替わりが
# あった（数値上の検算は成立するが意味的には不整合）。
# 案Aの実装では、最新 timestamp 群（複数行ある場合は群全体）を品質
# 検証の前に鮮度除外し、input_rows を「鮮度除外後に品質検証へ入る
# 行数」とするため:
# - 最新足が NaN / ±Inf / OHLC不正でも、その行は品質カウンタに
#   混入せず、有効な直前確定足は追加で除外されない。
# - 最新 timestamp の複数行は群全体が鮮度除外され、重複としては
#   計上されない。
# - NaT 行は鮮度除外に含めず、dropped_invalid_timestamp に計上する。
# - 最新より前の重複 timestamp は従来どおり keep="last" で処理する。
# - 検算式は全ケースで意味的にも成立し、学習対象行数は実際の処理
#   結果（Scanner は戻り値の行数、App は Target が付いた行数）と一致
#   する。正常系の期待値は案A導入前と同一（出力は変わらない）。
# ---------------------------------------------------------------------------

def _last_row_case_frames():
    cases = {}
    cases["normal"] = make_frame()

    f = make_frame()
    f.iloc[-1, f.columns.get_loc("Close")] = np.nan
    cases["nan"] = f

    f = make_frame()
    f.iloc[-1, f.columns.get_loc("High")] = np.inf
    cases["inf"] = f

    f = make_frame()
    f.iloc[-1, f.columns.get_loc("Low")] = -np.inf
    cases["neginf"] = f

    f = make_frame()
    f.iloc[-1, f.columns.get_loc("High")] = f.iloc[-1]["Low"] - 0.5
    cases["ohlc_invalid"] = f

    f = make_frame()
    dup = f.iloc[[-1]].copy()
    cl = dup.columns.get_loc("Close")
    hi = dup.columns.get_loc("High")
    dup.iloc[0, cl] = float(dup.iloc[0]["Close"]) + 0.03
    dup.iloc[0, hi] = max(float(dup.iloc[0]["High"]), float(dup.iloc[0]["Close"]))
    cases["duplicate"] = pd.concat([f, dup])

    # 最新 timestamp に複数行があり、うち1行が NaN。群全体が鮮度除外
    # され、NaN 行も品質カウンタには混入しない。
    f = make_frame()
    dup = f.iloc[[-1]].copy()
    dup.iloc[0, dup.columns.get_loc("Close")] = np.nan
    cases["latest_multi"] = pd.concat([f, dup])

    # 最新 timestamp より前に重複 timestamp がある。鮮度除外の対象では
    # なく、品質パイプラインで keep="last" により処理される。
    f = make_frame()
    dup = f.iloc[[350]].copy()
    cl = dup.columns.get_loc("Close")
    hi = dup.columns.get_loc("High")
    dup.iloc[0, cl] = float(dup.iloc[0]["Close"]) + 0.03
    dup.iloc[0, hi] = max(float(dup.iloc[0]["High"]), float(dup.iloc[0]["Close"]))
    cases["dup_early"] = pd.concat([f.iloc[:351], dup, f.iloc[351:]])

    f = make_frame()
    nat_row = f.iloc[[10]].copy()
    nat_row.index = pd.DatetimeIndex([pd.NaT])
    cases["nat"] = pd.concat([f, nat_row])
    return cases


def _assert_counter_profile(dq, expected):
    for key, value in expected.items():
        assert dq[key] == value, f"{key}: expected {value}, got {dq[key]}"
    assert dq["dropped_label_discontinuous"] == 0
    assert dq["dropped_label_other"] == 0
    assert dq["dropped_label_insufficient_future"] == 8
    assert_checksums(dq)


# Scanner: 特徴量除外 19 件（指標の立ち上がり）を前提とした実測値。
# 案Aでは最新 timestamp 群が鮮度除外されるため、最新足の品質不正は
# 品質カウンタに現れず、直前確定足も残る（feature_rows は正常系と同数）。
_SCANNER_LAST_ROW_EXPECTED = {
    "normal": {"input_rows": 499, "dropped_invalid_timestamp": 0,
               "dropped_missing_or_nonfinite": 0, "dropped_duplicate_timestamp": 0,
               "dropped_invalid_ohlc": 0, "dropped_feature_nan": 19,
               "feature_rows": 480, "trainable_rows": 472},
    "nan": {"input_rows": 499, "dropped_invalid_timestamp": 0,
            "dropped_missing_or_nonfinite": 0, "dropped_duplicate_timestamp": 0,
            "dropped_invalid_ohlc": 0, "dropped_feature_nan": 19,
            "feature_rows": 480, "trainable_rows": 472},
    "inf": {"input_rows": 499, "dropped_invalid_timestamp": 0,
            "dropped_missing_or_nonfinite": 0, "dropped_duplicate_timestamp": 0,
            "dropped_invalid_ohlc": 0, "dropped_feature_nan": 19,
            "feature_rows": 480, "trainable_rows": 472},
    "neginf": {"input_rows": 499, "dropped_invalid_timestamp": 0,
               "dropped_missing_or_nonfinite": 0,
               "dropped_duplicate_timestamp": 0, "dropped_invalid_ohlc": 0,
               "dropped_feature_nan": 19, "feature_rows": 480,
               "trainable_rows": 472},
    "ohlc_invalid": {"input_rows": 499, "dropped_invalid_timestamp": 0,
                     "dropped_missing_or_nonfinite": 0,
                     "dropped_duplicate_timestamp": 0, "dropped_invalid_ohlc": 0,
                     "dropped_feature_nan": 19, "feature_rows": 480,
                     "trainable_rows": 472},
    "duplicate": {"input_rows": 499, "dropped_invalid_timestamp": 0,
                  "dropped_missing_or_nonfinite": 0,
                  "dropped_duplicate_timestamp": 0, "dropped_invalid_ohlc": 0,
                  "dropped_feature_nan": 19, "feature_rows": 480,
                  "trainable_rows": 472},
    "latest_multi": {"input_rows": 499, "dropped_invalid_timestamp": 0,
                     "dropped_missing_or_nonfinite": 0,
                     "dropped_duplicate_timestamp": 0,
                     "dropped_invalid_ohlc": 0, "dropped_feature_nan": 19,
                     "feature_rows": 480, "trainable_rows": 472},
    "dup_early": {"input_rows": 500, "dropped_invalid_timestamp": 0,
                  "dropped_missing_or_nonfinite": 0,
                  "dropped_duplicate_timestamp": 1, "dropped_invalid_ohlc": 0,
                  "dropped_feature_nan": 19, "feature_rows": 480,
                  "trainable_rows": 472},
    "nat": {"input_rows": 500, "dropped_invalid_timestamp": 1,
            "dropped_missing_or_nonfinite": 0, "dropped_duplicate_timestamp": 0,
            "dropped_invalid_ohlc": 0, "dropped_feature_nan": 19,
            "feature_rows": 480, "trainable_rows": 472},
}

# App: 特徴量除外 15 件（17特徴量の立ち上がり）を前提とした実測値。
_APP_LAST_ROW_EXPECTED = {
    case: {
        **profile,
        "dropped_feature_nan": 15,
        "feature_rows": profile["feature_rows"] + 4,
        "trainable_rows": profile["trainable_rows"] + 4,
    }
    for case, profile in _SCANNER_LAST_ROW_EXPECTED.items()
}


def _check_scanner_last_row_case(monkeypatch, case):
    frame = _last_row_case_frames()[case]
    stub_scanner_download(monkeypatch, frame)
    out = scanner.bars("USDJPY=X", interval="1h", period="180d")
    assert out is not None
    dq = out.attrs["data_quality"]
    _assert_counter_profile(dq, _SCANNER_LAST_ROW_EXPECTED[case])
    assert dq["trainable_rows"] == len(out)


def _check_app_last_row_case(case):
    frame = _last_row_case_frames()[case]
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "60d", "1h")
    assert out is not None
    assert st.warnings == []
    dq = out.attrs["data_quality"]
    _assert_counter_profile(dq, _APP_LAST_ROW_EXPECTED[case])
    # 学習対象行数は、実際にラベルが付いた行数と一致する。
    assert int(out["Target"].notna().sum()) == dq["trainable_rows"]


def test_scanner_last_row_normal(monkeypatch):
    _check_scanner_last_row_case(monkeypatch, "normal")


def test_scanner_last_row_nan(monkeypatch):
    _check_scanner_last_row_case(monkeypatch, "nan")


def test_scanner_last_row_inf(monkeypatch):
    _check_scanner_last_row_case(monkeypatch, "inf")


def test_scanner_last_row_neginf(monkeypatch):
    _check_scanner_last_row_case(monkeypatch, "neginf")


def test_scanner_last_row_ohlc_invalid(monkeypatch):
    _check_scanner_last_row_case(monkeypatch, "ohlc_invalid")


def test_scanner_last_row_duplicate(monkeypatch):
    _check_scanner_last_row_case(monkeypatch, "duplicate")


def test_scanner_last_row_latest_multi(monkeypatch):
    _check_scanner_last_row_case(monkeypatch, "latest_multi")


def test_scanner_last_row_dup_early(monkeypatch):
    _check_scanner_last_row_case(monkeypatch, "dup_early")


def test_scanner_last_row_nat(monkeypatch):
    _check_scanner_last_row_case(monkeypatch, "nat")


def test_app_last_row_normal():
    _check_app_last_row_case("normal")


def test_app_last_row_nan():
    _check_app_last_row_case("nan")


def test_app_last_row_inf():
    _check_app_last_row_case("inf")


def test_app_last_row_neginf():
    _check_app_last_row_case("neginf")


def test_app_last_row_ohlc_invalid():
    _check_app_last_row_case("ohlc_invalid")


def test_app_last_row_duplicate():
    _check_app_last_row_case("duplicate")


def test_app_last_row_latest_multi():
    _check_app_last_row_case("latest_multi")


def test_app_last_row_dup_early():
    _check_app_last_row_case("dup_early")


def test_app_last_row_nat():
    _check_app_last_row_case("nat")


def test_scanner_invalid_latest_bar_keeps_previous_bar(monkeypatch):
    # 案Aの要点: 最新足が品質不正でも、有効な直前確定足が追加で
    # 除外されない（旧実装では鮮度除外が直前確定足に移っていた）。
    for case in ("nan", "inf", "neginf", "ohlc_invalid"):
        frame = _last_row_case_frames()[case]
        stub_scanner_download(monkeypatch, frame)
        out = scanner.bars("USDJPY=X", interval="1h", period="180d")
        assert out is not None
        # 直前確定足は特徴量フレームに残る。
        assert frame.index[-2] in out.attrs["latest_features"].index
        # 最新 timestamp は鮮度除外され、特徴量フレームには残らない。
        assert frame.index[-1] not in out.attrs["latest_features"].index


def test_app_invalid_latest_bar_keeps_previous_bar():
    for case in ("nan", "inf", "neginf", "ohlc_invalid"):
        frame = _last_row_case_frames()[case]
        ns, st = _app_namespace(lambda f=frame: f.copy())
        out = ns["load_and_process_data"]("USDJPY=X", "60d", "1h")
        assert out is not None
        assert st.warnings == []
        assert frame.index[-2] in out.index
        assert frame.index[-1] not in out.index


def test_scanner_min_bars_boundary_unchanged(monkeypatch):
    # 行数境界は案A導入前と同一: ダウンロード MIN_FEATURE_BARS+1 本
    # （= 221 本）で処理され、MIN_FEATURE_BARS 本（= 220 本）では停止
    # する。鮮度除外を前倒しした分、事前チェックの +1 本は不要になる。
    frame = make_frame(n=scanner.MIN_FEATURE_BARS)
    stub_scanner_download(monkeypatch, frame)
    assert scanner.bars("USDJPY=X", interval="1h", period="180d") is None
    frame = make_frame(n=scanner.MIN_FEATURE_BARS + 1)
    stub_scanner_download(monkeypatch, frame)
    out = scanner.bars("USDJPY=X", interval="1h", period="180d")
    assert out is not None
    dq = out.attrs["data_quality"]
    assert dq["input_rows"] == scanner.MIN_FEATURE_BARS
    assert dq["trainable_rows"] == len(out)
    assert_checksums(dq)


def test_app_min_bars_boundary_unchanged():
    # App の境界も従来どおり: ダウンロード 80 本では停止し、81 本で
    # 処理される（鮮度除外後の 80 本が下限判定にかかる）。
    frame = make_frame(n=80)
    ns, _st = _app_namespace(lambda f=frame: f.copy())
    assert ns["load_and_process_data"]("USDJPY=X", "60d", "1h") is None
    frame = make_frame(n=81)
    ns, _st = _app_namespace(lambda f=frame: f.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "60d", "1h")
    assert out is not None
    assert out.attrs["data_quality"]["input_rows"] == 80


# ---------------------------------------------------------------------------
# 追加レビュー対応（項目1）: 停止理由の記録とキャッシュ外表示（単体）
#
# キャッシュ対象のコアは停止理由を戻り値で返し、公開ラッパーが
# キャッシュの外側で記録・表示する。キャッシュヒット時の表示の実測は
# tests/test_stage1_app_cache.py（AppTest）側で行う。ここでは抽出により
# 次を確認する: 停止理由が理由コード付きで記録されること、取得失敗では
# 記録も警告も出ないこと（停止との区別）、成功時に記録が消えること、
# 同一描画内では同じ理由の警告が重複表示されないこと。
# ---------------------------------------------------------------------------

def test_app_stop_reason_recorded_for_tz_naive_daily():
    frame = make_frame()
    frame.index = frame.index.tz_localize(None)
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "2y", "1d")
    assert out is None
    reason = ns["get_load_stop_reason"]("USDJPY=X", "1d")
    assert reason is not None
    assert reason["code"] == "tz_unavailable"
    assert any("タイムゾーン" in w and "停止" in w for w in st.warnings)
    # 同じ呼び出しを繰り返しても、警告は同一描画内で1回だけ。
    out2 = ns["load_and_process_data"]("USDJPY=X", "2y", "1d")
    assert out2 is None
    assert len(st.warnings) == 1


def test_app_stop_reason_recorded_for_bad_index():
    frame = make_frame()
    frame.index = pd.RangeIndex(len(frame))
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "60d", "1h")
    assert out is None
    reason = ns["get_load_stop_reason"]("USDJPY=X", "1h")
    assert reason is not None
    assert reason["code"] == "timestamp_format_unconfirmed"
    assert any("時刻形式" in w and "停止" in w for w in st.warnings)


def test_app_download_failure_has_no_stop_reason():
    ns, st = _app_namespace(lambda: pd.DataFrame())
    out = ns["load_and_process_data"]("USDJPY=X", "2y", "1d")
    assert out is None
    # 取得失敗は停止ではない: 理由は記録されず、警告も出ない。
    assert ns["get_load_stop_reason"]("USDJPY=X", "1d") is None
    assert st.warnings == []


def test_app_stop_reason_cleared_on_success():
    naive = make_frame()
    naive.index = naive.index.tz_localize(None)
    aware = make_frame()
    frames = [naive, aware]
    ns, st = _app_namespace(lambda: frames.pop(0).copy())
    out = ns["load_and_process_data"]("USDJPY=X", "2y", "1d")
    assert out is None
    assert ns["get_load_stop_reason"]("USDJPY=X", "1d") is not None
    out = ns["load_and_process_data"]("USDJPY=X", "2y", "1d")
    assert out is not None
    assert ns["get_load_stop_reason"]("USDJPY=X", "1d") is None
    assert len(st.warnings) == 1
