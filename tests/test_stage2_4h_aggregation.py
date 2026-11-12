"""段階2（SYN-FX-017）App 4時間足の集約統一のテスト。

承認済み修正仕様書（2026-11-11）の段階2の実装に対する合成テスト。
実データ・実ネットワークは使わない（yfinance はスタブに差し替え、
ソケット接続はガードで禁止する）。

段階2の変更（App のみ。Scanner は正本として変更しない）:
- 4時間足の源データ（1時間足ダウンロード）の index を、集約前に
  UTC へ変換し、resample("4h", origin="start_day")（UTC の 00:00
  起点）で集約する。これにより App と Scanner で、同じ絶対時刻の
  データから同じ 4h 区間が生成される。
- 完全性判定を導入する: 区間ごとに元の 1時間足の本数を数え、
  4本そろった区間だけを採用する（Scanner の完全性判定と同じ考え方）。
  最終行の一律除外（iloc[:-1]）は 4h 経路では廃止し、この完全性
  判定に置き換える。
- 実行時タイムゾーン検査を 4h にも適用する: DatetimeIndex でない
  場合と tz-naive の場合は、UTC と推定せず、段階1と同じ停止理由の
  仕組み（コード付きの理由を戻り値で返し、ラッパーが記録・表示）
  で停止する。
- App MODEL_VERSION は段階2で 7 → 8（モデル再利用の仕組み自体は
  変更しない。版管理の確認は tests/test_stage1_input_validation.py
  のモデルテストで 7→8 に更新して行う）。
- C-1 カウンタ（data_quality）は引き続き 1時間足のみで、4h には
  付かない。特徴量・ラベル・品質ゲート・閾値・B-2・Future_*・
  purge は変更しない（B-2 の連続性規則は 4h に適用しない）。

境界（最低行数）の意味の変化:
- 段階2後の 4h 経路の最低行数判定（len(df) < 80 で停止）は、
  「完全な 4h 区間の本数」に対して行われる。したがって 320 本の
  1時間足（完全な 80 区間）で処理が始まり、316 本（79 区間）では
  停止する。不完全な最終区間は本数に数えない。
- 段階2前は「空でない区間の本数 − 1（最終行一律除外）」だったため、
  同じ 320 本でも 80 − 1 = 79 で停止していた（旧実装との差異の
  実測は段階2の測定記録に残し、PR 本文に記載する）。

段階2再レビュー対応（ChatGPT 独立レビューの保留2点への修正）:
- cutoff を Scanner と同一式で導入する: cutoff = 現在時刻（UTC）
  − 5分とし、終了時刻（index+1h）が cutoff 以前の 1時間足だけを
  集約対象に残し、採用条件にも区間の終了（index+4h <= cutoff）を
  明示する。これにより形成中の最終区間が App でも採用されなく
  なり、最終 4h 足の採用条件が Scanner と一致する。
- 集約前の重複 timestamp 整理（案b）: 4h 分岐で dropna の後・
  sort_index の前に、Scanner の共通整形と文字どおり同じ式で
  keep="last" の整理を行う。これにより完全性判定の本数が
  「異なる timestamp の本数」になり、集約値（OHLC・Volume）も
  Scanner と一致する。
- cutoff の境界は、pd.Timestamp.now を関数スコープの monkeypatch
  で固定する決定的テストで確認する（Scanner 実物と App 実物の
  双方に同じ時刻を与える）。復元は monkeypatch の undo にのみ
  任せ、手作業の退避・復元は行わない（実測で Timestamp.now が
  壊れることを確認済み）。
- MODEL_VERSION は 8 のまま据え置く。
"""

import ast
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

import scanner

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app.py"

OHLC = ["Open", "High", "Low", "Close"]
# App の特徴量ウォームアップで先頭から落ちる行数（段階1で実測した
# App の dropped_feature_nan = 15 と同じ指標計算のため、4h でも
# 先頭 15 区間が特徴量 NaN で落ちる）。
APP_FEATURE_WARMUP = 15
# Scanner の特徴量ウォームアップ（段階1で実測した Scanner の
# dropped_feature_nan = 19）。
SCANNER_FEATURE_WARMUP = 19


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """実ネットワーク接続を禁止する（スタブ以外の取得経路を塞ぐ）。"""

    def guard(*args, **kwargs):
        raise AssertionError("real network access attempted during stage2 tests")

    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)


# ---------------------------------------------------------------------------
# 共通フィクスチャ（段階1テストと同じ生成方式）
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


def stub_scanner_download(monkeypatch, frame):
    monkeypatch.setattr(yf, "download", lambda *a, **k: frame.copy())


def expected_4h_bins(frame):
    """Scanner の 4h 採用条件を素直に適用した期待区間系列。

    UTC 変換 → resample("4h", origin="start_day") → 元の 1時間足が
    4本そろった区間だけを採用する。Scanner 実物（bars() の戻り値の
    attrs["latest_features"]）と App の戻り値の双方を、この系列と
    突き合わせて検証する（ Scanner には現在時刻ベースの cutoff が
    あるが、合成フレームは過去の時刻のため cutoff の影響はない）。
    """
    df = frame.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC")
    df = (
        df.apply(pd.to_numeric, errors="coerce")
        .dropna(subset=OHLC)
        .sort_index()
    )
    counts = df["Close"].resample("4h", origin="start_day").count()
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample("4h", origin="start_day").agg(agg).dropna(subset=OHLC)
    complete = counts.reindex(out.index).fillna(0)
    return out.loc[complete == 4]


# ---------------------------------------------------------------------------
# App: AST 抽出による load_and_process_data の検証（段階1と同じ手法）
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


def _assert_suffix_of_expected(out, expected, warmup):
    """戻り値が期待区間系列の末尾側（ウォームアップ分を除く接尾辞）
    であり、OHLC が一致することを確認する。"""
    assert out is not None
    assert len(out) == len(expected) - warmup
    pd.testing.assert_index_equal(out.index, expected.index[warmup:])
    pd.testing.assert_frame_equal(out[OHLC], expected[OHLC].iloc[warmup:])


# ---------------------------------------------------------------------------
# App と Scanner の 4h 集約の一致（境界・本数・OHLC）
# ---------------------------------------------------------------------------

def test_app_4h_matches_scanner_utc(monkeypatch):
    frame = make_frame(n=1600, tz="UTC", seed=21)
    expected = expected_4h_bins(frame)
    assert len(expected) == 400

    stub_scanner_download(monkeypatch, frame)
    scan = scanner.bars("USDJPY=X", interval="4h", period="90d")
    assert scan is not None
    scan_series = scan.attrs["latest_features"]
    # Scanner 実物の集約系列（特徴量ウォームアップを除く接尾辞）。
    _assert_suffix_of_expected(scan_series, expected, SCANNER_FEATURE_WARMUP)

    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert st.warnings == []
    _assert_suffix_of_expected(out, expected, APP_FEATURE_WARMUP)

    # 両実装の共通部分で OHLC が完全一致する。
    pd.testing.assert_frame_equal(
        out.loc[scan_series.index, OHLC], scan_series[OHLC]
    )
    # 最終区間（完全）は両実装とも採用する（段階2で App の一律除外を
    # 廃止したため、App の最新足が Scanner と同じ区間になる）。
    last_bin = expected.index[-1]
    assert last_bin in out.index
    assert last_bin in scan_series.index
    assert out.loc[last_bin, "Close"] == scan_series.loc[last_bin, "Close"]


@pytest.mark.parametrize("tz", ["Asia/Tokyo", "America/New_York"])
def test_app_4h_matches_scanner_nonutc_tz(monkeypatch, tz):
    # 非 UTC のタイムゾーンでも、集約境界は UTC 基準で Scanner と
    # 一致する（段階2前は index のタイムゾーンの start_day 基準で
    # 境界がずれていた）。
    n = 1600 if tz == "Asia/Tokyo" else 1200
    frame = make_frame(n=n, tz=tz, seed=22)
    expected = expected_4h_bins(frame)
    assert len(expected) >= 250

    stub_scanner_download(monkeypatch, frame)
    scan = scanner.bars("USDJPY=X", interval="4h", period="90d")
    assert scan is not None
    scan_series = scan.attrs["latest_features"]
    _assert_suffix_of_expected(scan_series, expected, SCANNER_FEATURE_WARMUP)

    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert st.warnings == []
    _assert_suffix_of_expected(out, expected, APP_FEATURE_WARMUP)
    # 戻り値の index は UTC で、区間境界は UTC の 00:00 起点
    # （開始時刻の「時」が 4 の倍数）にそろう。
    assert str(out.index.tz) == "UTC"
    assert all(ts.hour % 4 == 0 for ts in out.index)
    pd.testing.assert_frame_equal(
        out.loc[scan_series.index, OHLC], scan_series[OHLC]
    )


def test_app_4h_gap_bin_excluded_by_both(monkeypatch):
    # 区間の途中に 1時間の欠損がある場合、その区間は元の足が
    # 3本しかないため、App・Scanner とも採用しない。
    frame = make_frame(n=1600, tz="UTC", seed=23)
    gap_ts = frame.index[402]  # 区間（400〜403 本目）の途中
    gapped = frame.drop(index=[gap_ts])
    expected = expected_4h_bins(gapped)
    gap_bin = frame.index[400]  # 欠損を含む区間の開始時刻（UTC 境界）
    assert gap_bin not in expected.index
    assert len(expected) == 399

    stub_scanner_download(monkeypatch, gapped)
    scan = scanner.bars("USDJPY=X", interval="4h", period="90d")
    assert scan is not None
    scan_series = scan.attrs["latest_features"]
    assert gap_bin not in scan_series.index
    _assert_suffix_of_expected(scan_series, expected, SCANNER_FEATURE_WARMUP)

    ns, _st = _app_namespace(lambda: gapped.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert gap_bin not in out.index
    _assert_suffix_of_expected(out, expected, APP_FEATURE_WARMUP)
    pd.testing.assert_frame_equal(
        out.loc[scan_series.index, OHLC], scan_series[OHLC]
    )


# ---------------------------------------------------------------------------
# 最終区間: 完全なら残し、不完全なら落とす
# ---------------------------------------------------------------------------

def test_app_4h_final_bin_complete_kept():
    frame = make_frame(n=400, tz="UTC", seed=24)  # 完全な 100 区間
    expected = expected_4h_bins(frame)
    assert len(expected) == 100
    ns, _st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    _assert_suffix_of_expected(out, expected, APP_FEATURE_WARMUP)
    # 最終区間が採用され、戻り値の最新足が系列の最終区間になる
    # （段階2前は一律除外で 1 区間古くなっていた）。
    assert out.index[-1] == expected.index[-1]


@pytest.mark.parametrize("extra_bars", [1, 2, 3])
def test_app_4h_final_bin_incomplete_dropped(extra_bars):
    # 最終区間が不完全（1〜3 本）な場合は、その区間を採用しない。
    frame = make_frame(n=400 + extra_bars, tz="UTC", seed=25)
    expected = expected_4h_bins(frame)
    assert len(expected) == 100
    ns, _st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    _assert_suffix_of_expected(out, expected, APP_FEATURE_WARMUP)
    partial_bin = frame.index[400]  # 不完全な最終区間の開始時刻
    assert partial_bin not in out.index
    assert out.index[-1] == expected.index[-1]


# ---------------------------------------------------------------------------
# タイムゾーン安全停止（4h）
# ---------------------------------------------------------------------------

def test_app_4h_tz_naive_stops_with_reason():
    frame = make_frame(n=1600, tz="UTC", seed=26)
    frame.index = frame.index.tz_localize(None)
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert out is None
    assert any("タイムゾーン" in w and "停止" in w for w in st.warnings)
    reason = ns["get_load_stop_reason"]("USDJPY=X", "4h")
    assert reason is not None
    assert reason["code"] == "tz_unavailable"


def test_app_4h_tz_stop_reason_cleared_after_success():
    # 停止理由は、次に正常に取得できた時点で解除される（段階1の
    # ラッパーの仕組みが 4h でも同じに働くことの確認）。
    naive = make_frame(n=1600, tz="UTC", seed=26)
    naive.index = naive.index.tz_localize(None)
    aware = make_frame(n=1600, tz="UTC", seed=26)
    frames = [naive, aware]
    ns, _st = _app_namespace(lambda: frames.pop(0).copy())
    assert ns["load_and_process_data"]("USDJPY=X", "90d", "4h") is None
    assert ns["get_load_stop_reason"]("USDJPY=X", "4h") is not None
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert out is not None
    assert ns["get_load_stop_reason"]("USDJPY=X", "4h") is None


def test_app_4h_non_datetime_index_stops():
    frame = make_frame(n=1600, tz="UTC", seed=27)
    frame.index = pd.RangeIndex(len(frame))
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert out is None
    reason = ns["get_load_stop_reason"]("USDJPY=X", "4h")
    assert reason is not None
    assert reason["code"] == "timestamp_format_unconfirmed"
    assert any("時刻形式" in w and "停止" in w for w in st.warnings)


def test_app_4h_has_no_data_quality_counters():
    # C-1 カウンタは引き続き 1時間足のみ。4h の戻り値に data_quality
    # は付かず、警告も出ない。
    frame = make_frame(n=1600, tz="UTC", seed=28)
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert out is not None
    assert "data_quality" not in out.attrs
    assert st.warnings == []


# ---------------------------------------------------------------------------
# 最低行数境界（80/81）の再確認: 完全な区間の本数で判定する
# ---------------------------------------------------------------------------

def test_app_4h_boundary_79_complete_bins_stops():
    # 316 本 = 完全な 79 区間 → 80 未満で停止する。
    frame = make_frame(n=316, tz="UTC", seed=29)
    assert len(expected_4h_bins(frame)) == 79
    ns, _st = _app_namespace(lambda: frame.copy())
    assert ns["load_and_process_data"]("USDJPY=X", "90d", "4h") is None


def test_app_4h_boundary_80_complete_bins_proceeds():
    # 320 本 = 完全な 80 区間 → 処理が始まる。段階2前は最終行の
    # 一律除外で 80 − 1 = 79 となり停止していたケースであり、
    # 境界の意味が「完全な区間の本数」に変わったことを固定する。
    frame = make_frame(n=320, tz="UTC", seed=29)
    expected = expected_4h_bins(frame)
    assert len(expected) == 80
    ns, _st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    _assert_suffix_of_expected(out, expected, APP_FEATURE_WARMUP)


def test_app_4h_boundary_partial_final_bin_not_counted():
    # 不完全な最終区間は境界の本数に数えない: 79 完全 + 不完全では
    # 停止し、80 完全 + 不完全では処理が始まる。
    frame = make_frame(n=318, tz="UTC", seed=30)  # 79 完全 + 2 本
    assert len(expected_4h_bins(frame)) == 79
    ns, _st = _app_namespace(lambda: frame.copy())
    assert ns["load_and_process_data"]("USDJPY=X", "90d", "4h") is None

    frame = make_frame(n=322, tz="UTC", seed=30)  # 80 完全 + 2 本
    expected = expected_4h_bins(frame)
    assert len(expected) == 80
    ns, _st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    _assert_suffix_of_expected(out, expected, APP_FEATURE_WARMUP)


# ---------------------------------------------------------------------------
# cutoff の決定的境界テスト（段階2再レビュー対応・T1）
#
# pd.Timestamp.now を関数スコープの monkeypatch で固定値に差し替え、
# Scanner 実物と App 実物の双方に同じ時刻を与える。差し替えはクラス
# 単位で効くため、両実装の時刻源（pd.Timestamp.now のみ）が同じ固定
# 値になり、実行時の実時刻に依存しない。復元は monkeypatch の undo
# にのみ任せる（手作業で退避・復元・削除を行うと Timestamp.now が
# 壊れることを実測済み）。
# ---------------------------------------------------------------------------

def _freeze_now(monkeypatch, fixed):
    """pd.Timestamp.now を固定値に差し替える（関数スコープ限定）。"""
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda cls, tz=None: fixed)
    )


def _run_both_4h(monkeypatch, frame):
    """同じフレームを Scanner 実物と App 実物の双方に与え、集約系列
    （Scanner は attrs["latest_features"]、App は戻り値）を返す。"""
    stub_scanner_download(monkeypatch, frame)
    scan = scanner.bars("USDJPY=X", interval="4h", period="90d")
    assert scan is not None
    scan_series = scan.attrs["latest_features"]
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert st.warnings == []
    return scan_series, out


def _assert_both_agree_on_bins(scan_series, out, adopted_last_bin):
    """両実装の採用区間が一致し、最新の採用区間が期待どおりである
    ことと、共通部分の OHLC が完全一致することを確認する。"""
    assert out is not None
    # 両実装とも同じ採用系列の接尾辞（ウォームアップ差は一定）。
    assert len(out) - len(scan_series) == (
        SCANNER_FEATURE_WARMUP - APP_FEATURE_WARMUP
    )
    assert out.index[-1] == scan_series.index[-1] == adopted_last_bin
    pd.testing.assert_frame_equal(
        out.loc[scan_series.index, OHLC], scan_series[OHLC]
    )


def test_cutoff_boundary_end_equals_cutoff_adopted(monkeypatch):
    # ケース(1): 最終 1h 足の終了時刻が cutoff と一致する場合、その足
    # は採用対象（事前除外の条件は <=）となり、最終区間は両実装とも
    # 採用する。FIXED = E + 5分 → cutoff == E。
    frame = make_frame(n=1600, tz="UTC", seed=21)
    end = frame.index[-1] + pd.Timedelta(hours=1)
    _freeze_now(monkeypatch, end + pd.Timedelta(minutes=5))
    scan_series, out = _run_both_4h(monkeypatch, frame)
    final_bin = frame.index[-4]
    assert final_bin in scan_series.index
    assert final_bin in out.index
    _assert_both_agree_on_bins(scan_series, out, final_bin)


def test_cutoff_boundary_end_one_second_after_excluded(monkeypatch):
    # ケース(2): 最終 1h 足の終了時刻が cutoff の1秒後の場合、その足
    # は事前除外で落ち、最終区間は3本となり両実装とも採用しない。
    # FIXED = E + 5分 − 1秒 → cutoff == E − 1秒。
    frame = make_frame(n=1600, tz="UTC", seed=21)
    end = frame.index[-1] + pd.Timedelta(hours=1)
    _freeze_now(
        monkeypatch, end + pd.Timedelta(minutes=5) - pd.Timedelta(seconds=1)
    )
    scan_series, out = _run_both_4h(monkeypatch, frame)
    final_bin = frame.index[-4]
    assert final_bin not in scan_series.index
    assert final_bin not in out.index
    _assert_both_agree_on_bins(scan_series, out, frame.index[-8])


def test_cutoff_boundary_end_one_second_before_adopted(monkeypatch):
    # ケース(3): 最終 1h 足の終了時刻が cutoff の1秒前の場合、その足
    # は採用対象となり、最終区間は両実装とも採用する。
    # FIXED = E + 5分 + 1秒 → cutoff == E + 1秒。
    frame = make_frame(n=1600, tz="UTC", seed=21)
    end = frame.index[-1] + pd.Timedelta(hours=1)
    _freeze_now(
        monkeypatch, end + pd.Timedelta(minutes=5) + pd.Timedelta(seconds=1)
    )
    scan_series, out = _run_both_4h(monkeypatch, frame)
    final_bin = frame.index[-4]
    assert final_bin in scan_series.index
    assert final_bin in out.index
    _assert_both_agree_on_bins(scan_series, out, final_bin)


# ---------------------------------------------------------------------------
# cutoff 適用後の80区間境界（段階2再レビュー対応・T2、App のみ）
#
# 80区間の判定（len(df) < 80）は集約ブロックの後にあるため、cutoff
# 適用後に完全な区間が何区間残るかで開始・停止が決まることを固定
# する。Scanner の最低本数は生の本数基準で位置づけが異なるため、
# 確認対象は App のみ。
# ---------------------------------------------------------------------------

def test_cutoff_boundary_80_bins_after_cutoff_proceeds(monkeypatch):
    # 324 本（81 区間分）のうち最終区間の4本目が cutoff で落ちると、
    # 最終区間は3本で不採用となり、完全な区間はちょうど80 → 開始。
    frame = make_frame(n=324, tz="UTC", seed=31)
    end = frame.index[-1] + pd.Timedelta(hours=1)
    _freeze_now(
        monkeypatch, end + pd.Timedelta(minutes=5) - pd.Timedelta(seconds=1)
    )
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert st.warnings == []
    assert out is not None
    assert len(out) == 80 - APP_FEATURE_WARMUP
    assert out.index[-1] == frame.index[316]  # 80 番目の区間の開始
    assert frame.index[320] not in out.index  # 不完全な最終区間


def test_cutoff_boundary_79_bins_after_cutoff_stops(monkeypatch):
    # 320 本（完全80区間）の最終区間の4本目が cutoff で落ちると、
    # 完全な区間は79 → 停止する。cutoff は停止理由を生まないため、
    # 理由は記録されず警告も出ない（既存の不足扱い）。
    frame = make_frame(n=320, tz="UTC", seed=31)
    end = frame.index[-1] + pd.Timedelta(hours=1)
    _freeze_now(
        monkeypatch, end + pd.Timedelta(minutes=5) - pd.Timedelta(seconds=1)
    )
    ns, st = _app_namespace(lambda: frame.copy())
    out = ns["load_and_process_data"]("USDJPY=X", "90d", "4h")
    assert out is None
    assert st.warnings == []
    assert ns["get_load_stop_reason"]("USDJPY=X", "4h") is None


# ---------------------------------------------------------------------------
# 重複 timestamp（段階2再レビュー対応・T3、案b）
#
# 過去フレーム（実時刻では cutoff 不活性）のため時刻固定は不要。
# ---------------------------------------------------------------------------

def test_duplicate_timestamps_three_distinct_bin_rejected(monkeypatch):
    # ケースA: 1区間が異なる timestamp 3つ＋複製1行（行数は4）の
    # 場合、その区間は両実装とも採用しない。修正前の App は行数4で
    # 誤採用していたケース。
    base = make_frame(n=1600, tz="UTC", seed=32)
    bin_start = base.index[400]
    # 区間（400〜403 本目）の3本目（402）を欠落させ、2本目（401）の
    # 複製を直後に挿入する。
    frame = pd.concat([base.iloc[:402], base.iloc[[401]], base.iloc[403:]])
    scan_series, out = _run_both_4h(monkeypatch, frame)
    assert bin_start not in scan_series.index
    assert bin_start not in out.index
    # 採用区間の総数は 399 で、両実装とも最終区間まで一致する。
    assert len(out) == 399 - APP_FEATURE_WARMUP
    assert len(scan_series) == 399 - SCANNER_FEATURE_WARMUP
    _assert_both_agree_on_bins(scan_series, out, base.index[-4])


def test_duplicate_timestamps_four_distinct_bin_adopted(monkeypatch):
    # ケースB: 1区間が異なる timestamp 4つ＋複製1行（行数は5）の
    # 場合、両実装とも採用し、集約値は keep="last" で残った行（値
    # を変えた複製行）から計算されて一致する。修正前の App は行数
    # 5で不採用にしていたケース。
    base = make_frame(n=1600, tz="UTC", seed=32)
    bin_start = base.index[400]
    dup = base.iloc[[401]].copy()
    bin_high = float(base.iloc[400:404]["High"].max()) + 1.0
    bin_low = float(base.iloc[400:404]["Low"].min()) - 1.0
    dup.iloc[0, dup.columns.get_loc("Close")] = (
        float(dup.iloc[0]["Close"]) + 0.05
    )
    dup.iloc[0, dup.columns.get_loc("High")] = bin_high
    dup.iloc[0, dup.columns.get_loc("Low")] = bin_low
    dup.iloc[0, dup.columns.get_loc("Volume")] = 2_000_000.0
    # 複製は元の行の直後（ダウンロード順で後）に挿入し、keep="last"
    # で複製側が残るようにする。
    frame = pd.concat([base.iloc[:402], dup, base.iloc[402:]])
    scan_series, out = _run_both_4h(monkeypatch, frame)
    assert bin_start in scan_series.index
    assert bin_start in out.index
    # keep="last" の集約値: 区間の High・Low は複製行の値、Volume は
    # 複製行の 2,000,000 を含む4本分（5,000,000）になる。二重計上
    # （6,000,000）でも keep="first"（4,000,000）でもない。
    assert out.loc[bin_start, "High"] == bin_high
    assert out.loc[bin_start, "Low"] == bin_low
    assert out.loc[bin_start, "Volume"] == 5_000_000.0
    assert scan_series.loc[bin_start, "High"] == bin_high
    assert scan_series.loc[bin_start, "Low"] == bin_low
    _assert_both_agree_on_bins(scan_series, out, base.index[-4])
