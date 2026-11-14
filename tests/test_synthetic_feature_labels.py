"""旧AIFX（my-fx-ai）App / Scanner の特徴量・ラベル生成に対する隔離合成テスト。

目的（Notion「旧AIFX読み取り専用監査」ページの
「Muse実施依頼：隔離合成テスト実装・実行（ユーザー承認済み、2026-11-11）」）:
- 現行 main を基準に、App（app.py）と Scanner（scanner.py）の特徴量・
  ラベル生成・品質検証の挙動を、決定論的な合成 DataFrame で再検証する。
- 本番コード（app.py / scanner.py / 既存テスト / Workflow / requirements）は
  一切変更しない。本ファイルは新規追加のテスト専用ファイルである。
- 合成テストの PASS は、実市場での予測精度・利益・校正性能の証明ではない。

実装方法:
- Scanner は scanner.add_indicators() / scanner.bars() を直接対象にし、
  yfinance.download を決定論的な合成 DataFrame 返却 stub に差し替える。
- App は app.py 全体を import すると Streamlit UI が実行されるため、
  AST で対象関数定義だけを抽出し、必要な依存（numpy / pandas /
  ダウンロード stub / flatten_yf_df / 定数）を明示した namespace で実行する。
  これは元関数の「抽出実行」であり、Streamlit の cache ラッパ等は含まない。
  - 現行 main では @st.cache_data が内部関数 _load_and_process_data_cached
    に付いている。AST抽出時はデコレータを取り除き、公開ラッパーも含めて
    実処理を実行する。
  - 抽出した各関数のソース断片 SHA-256 と AST dump の SHA-256 は
    EXTRACTION_INFO に記録し、レポートへ残す。
- ネットワークは fail-closed: autouse fixture が socket / DNS / urllib の
  入口を遮断し、接続試行を記録する。遮断対象は接続系（connect /
  connect_ex / create_connection）、送信系（send / sendall / sendto /
  sendmsg / sendmsg_af / sendfile)、名前解決系（getaddrinfo /
  gethostbyname / gethostbyname_ex / gethostbyaddr / getnameinfo)、
  urllib（urlopen / urlretrieve)。requests / http.client は内部で
  socket.create_connection と getaddrinfo を使うため、これらで遮断される。
  ガード自体の動作確認テスト（test_network_guard_blocks_*）と、suite
  終了時に試行 0 件を確認するテスト
  （test_network_no_attempts_during_suite）を置く。
"""

import ast
import hashlib
import inspect
import io
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scanner  # noqa: E402

UTC = "UTC"
START = pd.Timestamp("2026-01-05 00:00", tz=UTC)  # Monday


# =========================================================
# ネットワーク遮断ガード（fail-closed）
# =========================================================

NETWORK_ATTEMPTS = []


_GUARD_MESSAGE = "network access blocked by synthetic-test guard"


def _record_and_block(kind, detail):
    NETWORK_ATTEMPTS.append((kind, detail))
    raise OSError(_GUARD_MESSAGE)


class _GuardedSocket(socket.socket):
    """接続系・送信系メソッドを遮断する socket。生成自体は許可する。

    送信系（send / sendall / sendto / sendmsg / sendmsg_af / sendfile）も
    実 syscall の前に例外で止める。connect が遮断されるため通常は接続済み
    ソケット自体が作れないが、継承 fd 等の想定外経路でも送信できないよう
    多層に遮断する。
    """

    def connect(self, address):
        _record_and_block("socket.connect", repr(address))

    def connect_ex(self, address):
        _record_and_block("socket.connect_ex", repr(address))

    def send(self, data, flags=0):
        _record_and_block("socket.send", repr(len(data)))

    def sendall(self, data, flags=0):
        _record_and_block("socket.sendall", repr(len(data)))

    def sendto(self, data, *args):
        address = args[-1] if args else None
        _record_and_block("socket.sendto", repr(address))

    def sendmsg(self, buffers, ancdata=(), flags=0):
        _record_and_block("socket.sendmsg", repr(len(buffers)))

    def sendmsg_af(self, af, buffers, ancdata=(), flags=0):
        _record_and_block("socket.sendmsg_af", repr(af))

    def sendfile(self, file, offset=0, count=None):
        _record_and_block("socket.sendfile", repr(offset))


def _blocked_create_connection(address, *args, **kwargs):
    _record_and_block("socket.create_connection", repr(address))


def _blocked_getaddrinfo(host, *args, **kwargs):
    _record_and_block("socket.getaddrinfo", repr(host))


def _blocked_gethostbyaddr(ip_address, *args, **kwargs):
    _record_and_block("socket.gethostbyaddr", repr(ip_address))


def _blocked_getnameinfo(sockaddr, *args, **kwargs):
    _record_and_block("socket.getnameinfo", repr(sockaddr))


def _blocked_urlopen(url, *args, **kwargs):
    _record_and_block("urllib.request.urlopen", repr(url))


def _blocked_urlretrieve(url, *args, **kwargs):
    _record_and_block("urllib.request.urlretrieve", repr(url))


@pytest.fixture(autouse=True)
def _network_guard(monkeypatch):
    monkeypatch.setattr(socket, "socket", _GuardedSocket)
    monkeypatch.setattr(socket, "create_connection", _blocked_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", _blocked_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname_ex", _blocked_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyaddr", _blocked_gethostbyaddr)
    monkeypatch.setattr(socket, "getnameinfo", _blocked_getnameinfo)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked_urlopen)
    monkeypatch.setattr(urllib.request, "urlretrieve", _blocked_urlretrieve)
    yield


def test_network_guard_blocks_and_records():
    """ガード自体の動作確認: 接続と DNS は失敗し、試行が記録される。"""
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    with pytest.raises(OSError):
        socket.getaddrinfo("example.com", 443)
    assert len(NETWORK_ATTEMPTS) >= 2
    NETWORK_ATTEMPTS.clear()  # 自己確認の試行は証跡から除く


def test_network_guard_blocks_send_paths():
    """ガード自体の動作確認: 送信系メソッドは実送信前に失敗し記録される。

    ここで発生させる試行はガード検証専用であり、記録は検証後に消去する
    （suite 本体の「試行 0 件」の証跡には含めない）。
    """
    start = len(NETWORK_ATTEMPTS)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(OSError):
            sock.send(b"x")
        with pytest.raises(OSError):
            sock.sendall(b"x")
        with pytest.raises(OSError):
            sock.sendto(b"x", ("127.0.0.1", 9))
        with pytest.raises(OSError):
            sock.sendmsg([b"x"])
        with pytest.raises(OSError):
            sock.sendmsg_af(socket.AF_INET, [b"x"])
        with pytest.raises(OSError):
            sock.sendfile(io.BytesIO(b"x"))
    finally:
        sock.close()
    kinds = [kind for kind, _ in NETWORK_ATTEMPTS[start:]]
    assert kinds == [
        "socket.send",
        "socket.sendall",
        "socket.sendto",
        "socket.sendmsg",
        "socket.sendmsg_af",
        "socket.sendfile",
    ]
    NETWORK_ATTEMPTS.clear()  # 自己確認の試行は証跡から除く


def test_network_guard_blocks_additional_entry_points():
    """ガード自体の動作確認: 逆引き・名前情報・urlretrieve も失敗し記録される。

    記録は検証後に消去する（suite 本体の証跡には含めない）。
    """
    start = len(NETWORK_ATTEMPTS)
    with pytest.raises(OSError):
        socket.gethostbyaddr("127.0.0.1")
    with pytest.raises(OSError):
        socket.getnameinfo(("127.0.0.1", 9), 0)
    with pytest.raises(OSError):
        urllib.request.urlopen("http://127.0.0.1/")
    with pytest.raises(OSError):
        urllib.request.urlretrieve("http://127.0.0.1/", "/tmp/never-downloaded")
    kinds = [kind for kind, _ in NETWORK_ATTEMPTS[start:]]
    assert kinds == [
        "socket.gethostbyaddr",
        "socket.getnameinfo",
        "urllib.request.urlopen",
        "urllib.request.urlretrieve",
    ]
    assert not os.path.exists("/tmp/never-downloaded")
    NETWORK_ATTEMPTS.clear()  # 自己確認の試行は証跡から除く


# =========================================================
# 合成 fixture（決定論的系列。固定 seed / UTC-aware index / 固定 OHLCV）
# =========================================================


def _idx(n, start=START):
    return pd.date_range(start, periods=n, freq="1h", tz=UTC)


def fixture_f0_constant(n=240):
    """F0: Close 一定、High/Low 一定幅。"""
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 100.010,
            "Low": 99.990,
            "Close": 100.0,
            "Volume": 100.0,
        },
        index=_idx(n),
    )


def fixture_f1_rising(n=260):
    """F1: 増分が周期的に 0.001/0.002/0.004 の正値系列（全増分は正、一定ではない）。"""
    inc = np.resize(np.array([0.001, 0.002, 0.004]), n - 1)
    close = 100.0 + np.concatenate(([0.0], np.cumsum(inc)))
    open_ = np.concatenate(([100.0], close[:-1]))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": close + 0.006,
            "Low": open_ - 0.006,
            "Close": close,
            "Volume": 100.0,
        },
        index=_idx(n),
    )


def fixture_f2_falling(n=260):
    """F2: F1 の増分符号を負にした系列（価格は正値を保つ）。"""
    inc = np.resize(np.array([0.001, 0.002, 0.004]), n - 1)
    close = 102.0 - np.concatenate(([0.0], np.cumsum(inc)))
    open_ = np.concatenate(([102.0], close[:-1]))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": open_ + 0.006,
            "Low": close - 0.006,
            "Close": close,
            "Volume": 100.0,
        },
        index=_idx(n),
    )


def fixture_mixed(n=280, start=START):
    """増減混在の決定論的系列。RSI 等が定義でき、ATR は約 0.01。"""
    pattern = np.array(
        [0.004, 0.002, -0.003, 0.001, -0.002, 0.003, -0.001, 0.002]
    )
    inc = np.resize(pattern, n - 1)
    close = 100.0 + np.concatenate(([0.0], np.cumsum(inc)))
    open_ = np.concatenate(([100.0], close[:-1]))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + 0.004,
            "Low": np.minimum(open_, close) - 0.004,
            "Close": close,
            "Volume": 100.0,
        },
        index=_idx(n, start),
    )


def fixture_seeded_walk(n, start=START, seed=42):
    """4h テスト用: seed 固定ランダムウォーク（周期パターンだと 4h 集約で
    増減が偏り RSI が定義できなくなるため）。"""
    rng = np.random.default_rng(seed)
    inc = rng.normal(0, 0.003, n - 1)
    close = 100.0 + np.concatenate(([0.0], np.cumsum(inc)))
    open_ = np.concatenate(([100.0], close[:-1]))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + 0.004,
            "Low": np.minimum(open_, close) - 0.004,
            "Close": close,
            "Volume": 100.0,
        },
        index=_idx(n, start),
    )


FIXTURE_BUILDERS = {
    "fixture_f0_constant": fixture_f0_constant,
    "fixture_f1_rising": fixture_f1_rising,
    "fixture_f2_falling": fixture_f2_falling,
    "fixture_mixed": fixture_mixed,
    "fixture_seeded_walk": fixture_seeded_walk,
}


def normalize_ohlc(frame):
    """Keep engineered synthetic candles valid after changing Open/High/Low/Close."""
    frame = frame.copy()
    frame["High"] = frame[["High", "Open", "Close"]].max(axis=1)
    frame["Low"] = frame[["Low", "Open", "Close"]].min(axis=1)
    return frame


def fixture_source_hashes():
    """入力データ生成コードの hash（実施時の共通記録用）。"""
    return {
        name: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
        for name, fn in FIXTURE_BUILDERS.items()
    }


# =========================================================
# yfinance.download の決定論的 stub
# =========================================================


class YFStub:
    def __init__(self, frame=None):
        self.frame = frame
        self.calls = []

    def download(self, ticker, **kwargs):
        self.calls.append((ticker, kwargs))
        return self.frame.copy()


def run_scanner_bars(monkeypatch, frame, interval="1h"):
    stub = YFStub(frame)
    monkeypatch.setattr(scanner.yf, "download", stub.download)
    return scanner.bars("SYNTH=X", interval, "60d")


# =========================================================
# App（app.py）の AST 抽出実行ハーネス
# =========================================================

EXTRACTION_INFO = {}


def _extract_app_module(func_names, assign_names):
    """app.py から対象の関数定義・代入だけを AST 抽出して Module を返す。"""
    src = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            segment = ast.get_source_segment(src, node)
            EXTRACTION_INFO[f"app.py:{node.name}"] = {
                "lineno": node.lineno,
                "source_sha256": hashlib.sha256(segment.encode()).hexdigest(),
                "ast_dump_sha256": hashlib.sha256(
                    ast.dump(node).encode()
                ).hexdigest(),
                "decorators": [ast.unparse(d) for d in node.decorator_list],
            }
            node.decorator_list = []  # Streamlit cache 等のデコレータは実行しない
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in assign_names for t in node.targets
        ):
            nodes.append(node)
    mod = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    return mod


def _extract_scanner_nested(outer_name, inner_names):
    """scanner.py の関数内包関数（仮想損益など）を AST 抽出して Module を返す。"""
    src = (REPO_ROOT / "scanner.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == outer_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in inner_names:
                    segment = ast.get_source_segment(src, sub)
                    EXTRACTION_INFO[f"scanner.py:{outer_name}.{sub.name}"] = {
                        "lineno": sub.lineno,
                        "source_sha256": hashlib.sha256(
                            segment.encode()
                        ).hexdigest(),
                        "ast_dump_sha256": hashlib.sha256(
                            ast.dump(sub).encode()
                        ).hexdigest(),
                        "decorators": [],
                    }
                    nodes.append(sub)
    mod = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    return mod


# --- load_and_process_data 用 namespace（yf は stub に差し替え） ---
_yf_app = YFStub()
_app_load_mod = _extract_app_module(
    {"flatten_yf_df", "_load_and_process_data_cached", "load_and_process_data"},
    {"FEATURE_COLUMNS", "DEFAULT_SPREAD_PIPS", "LOOKAHEAD_BARS"},
)
class _LoadStreamlitStub:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(str(message))


APP_LOAD_NS = {
    "np": np,
    "pd": pd,
    "yf": _yf_app,
    "st": _LoadStreamlitStub(),
    "_LOAD_STOP_REASONS": {},
    "_LOAD_STOP_WARNED": set(),
}
exec(compile(_app_load_mod, "app.py[ast-extract]", "exec"), APP_LOAD_NS)
APP_FEATURE_COLUMNS = APP_LOAD_NS["FEATURE_COLUMNS"]


def run_app_load(frame, ticker="USDJPY=X", interval="1h"):
    _yf_app.frame = frame
    return APP_LOAD_NS["load_and_process_data"](ticker, "60d", interval)


# --- analyze_signal_with_backtest 用 namespace（st は fake） ---
_app_analyze_mod = _extract_app_module(
    {"analyze_signal_with_backtest", "_model_filename"},
    {"FEATURE_COLUMNS", "LOOKAHEAD_BARS", "MODEL_VERSION"},
)


class _FakeStreamlit:
    """analyze_signal_with_backtest が使う st.session_state / st.warning の代替。"""

    def __init__(self):
        self.session_state = {}
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


def fresh_app_analyze():
    """fake st を差し替えた新しい namespace で analyze 関数を作り直す。"""
    import joblib
    import time
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight

    fake_st = _FakeStreamlit()
    ns = {
        "np": np,
        "pd": pd,
        "os": os,
        "time": time,
        "joblib": joblib,
        "datetime": datetime,
        "JST": ZoneInfo("Asia/Tokyo"),
        "st": fake_st,
        "RandomForestClassifier": RandomForestClassifier,
        "DummyClassifier": DummyClassifier,
        "compute_sample_weight": compute_sample_weight,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
    }
    exec(compile(_app_analyze_mod, "app.py[ast-extract-analyze]", "exec"), ns)
    return ns["analyze_signal_with_backtest"], fake_st


# --- Scanner 仮想損益（walk_forward_validation 内包関数）の抽出 ---
_pnl_mod = _extract_scanner_nested(
    "walk_forward_validation", {"summarize_trade_results", "simulated_pnl"}
)
PNL_NS = {
    "np": np,
    "pd": pd,
    "safe_float": scanner.safe_float,
    "LOOKAHEAD_BARS": scanner.LOOKAHEAD_BARS,
    "SIMULATED_ROUND_TRIP_SLIPPAGE_PIPS": scanner.SIMULATED_ROUND_TRIP_SLIPPAGE_PIPS,
    "SIMULATED_SWAP_STRESS_COST_PIPS": scanner.SIMULATED_SWAP_STRESS_COST_PIPS,
}
exec(compile(_pnl_mod, "scanner.py[ast-extract-nested]", "exec"), PNL_NS)
SIMULATED_PNL = PNL_NS["simulated_pnl"]


def test_extraction_integrity():
    """抽出ハーネスの記録: 対象関数が app.py / scanner.py の原文どおりに
    抽出され、hash が再現可能であることを確認する（抽出実行の明示）。"""
    for key in (
        "app.py:flatten_yf_df",
        "app.py:_load_and_process_data_cached",
        "app.py:load_and_process_data",
        "app.py:analyze_signal_with_backtest",
        "scanner.py:walk_forward_validation.summarize_trade_results",
        "scanner.py:walk_forward_validation.simulated_pnl",
    ):
        assert key in EXTRACTION_INFO, key
        assert len(EXTRACTION_INFO[key]["source_sha256"]) == 64
    # 現行 main では内部処理関数のStreamlit cacheデコレータを抽出時に除去。
    assert EXTRACTION_INFO["app.py:_load_and_process_data_cached"]["decorators"] == [
        "st.cache_data(ttl=60, show_spinner=False)"
    ]
    hashes = fixture_source_hashes()
    assert all(len(v) == 64 for v in hashes.values())
    print("EXTRACTION_INFO:", json.dumps(EXTRACTION_INFO, indent=1))
    print("FIXTURE_SOURCE_SHA256:", json.dumps(hashes, indent=1))


# =========================================================
# 第1群: 特徴量・warm-up・未来情報（SYN-FX-001〜006）
# =========================================================


def _first_valid_pos(series):
    loc = series.first_valid_index()
    return None if loc is None else series.index.get_loc(loc)


def test_syn_fx_001_constant_series_safety(monkeypatch):
    """001: Close 一定・High/Low 一定幅。ゼロ除算・Inf が発生せず、定義不能な
    比率は欠損として残る。App の neutral fill と Scanner の欠損方針の差を記録。"""
    f0 = fixture_f0_constant()

    out = scanner.add_indicators(f0)
    numeric = out.select_dtypes("number")
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()
    # Scanner: ゼロ loss / ゼロ幅では RSI・ADX・BB_PctB・CCI は全て欠損のまま。
    for col in ("RSI", "ADX", "BB_PctB", "CCI"):
        assert out[col].isna().all(), col
    # ATR は TR=0.02 一定で、Wilder EWM(min_periods=14) のため位置 13 から有効。
    assert _first_valid_pos(out["ATR"]) == 13
    assert out["ATR"].dropna().iloc[0] == pytest.approx(0.02)
    # Stoch_K は (Close-Low)/(High-Low) = 0.5 で定義できる。
    assert out["Stoch_K"].dropna().iloc[0] == pytest.approx(50.0)
    assert (out["Vol_Ratio"].dropna() == 1.0).all()
    # Scanner bars(): 特徴量 dropna で空になり None を返す。
    assert run_scanner_bars(monkeypatch, f0) is None

    # App: BB_PctB が全行 NaN（標準偏差 0）のため最終 dropna で 0 行になる。
    # None ではなく空 DataFrame が返る、という現行挙動を記録する。
    res = run_app_load(f0)
    assert res is None or len(res) == 0

    # App の neutral fill の確認: 方向移動が無い系列（High/Low 一定幅・Close のみ
    # 変動）では DX が定義できず ADX は fillna(20) になる。
    n = 260
    close = fixture_mixed(n)["Close"]
    zero_dm = pd.DataFrame(
        {
            "Open": close,
            "High": 101.0,
            "Low": 99.0,
            "Close": close,
            "Volume": 100.0,
        },
        index=_idx(n),
    )
    res_zdm = run_app_load(zero_dm)
    assert res_zdm is not None and len(res_zdm) > 0
    assert (res_zdm["ADX"] == 20).all()


def test_syn_fx_002_monotonic_direction(monkeypatch):
    """002: 単調上昇/下降系列。RSI/ROC 等の符号・方向が期待と整合する。
    App/Scanner の数値一致は要求しない。"""
    f1, f2 = fixture_f1_rising(), fixture_f2_falling()

    s1 = scanner.add_indicators(f1)
    # Scanner RSI: 上昇系列は avg_loss=0 が NaN 化され、全行欠損になる現行挙動。
    assert s1["RSI"].isna().all()
    assert (s1["Return_1"].iloc[1:] > 0).all()
    assert (s1["ROC"].iloc[10:] > 0).all()

    s2 = scanner.add_indicators(f2)
    # 下降系列は gain=0 のため RSI=0 になる。
    assert _first_valid_pos(s2["RSI"]) == 14
    assert (s2["RSI"].dropna() == 0).all()
    assert (s2["ROC"].iloc[10:] < 0).all()

    # App RSI: loss=0 は NaN 化の後に 50 補完されるため、完全上昇でも 100 でなく 50。
    a1 = run_app_load(f1)
    assert a1 is not None and len(a1) > 0
    assert (a1["RSI"] == 50).all()
    assert (a1["ROC"] > 0).all()
    a2 = run_app_load(f2)
    assert (a2["RSI"] == 0).all()
    assert (a2["ROC"] < 0).all()


def test_syn_fx_003_zero_volume(monkeypatch):
    """003: Volume 全ゼロ。Scanner Vol_Ratio は 1.0 へフォールバック。
    App Vol_Ratio は Volume 由来ではなく ATR 由来であることを示す。"""
    zero_vol = fixture_mixed()
    zero_vol["Volume"] = 0.0
    s_zero = scanner.add_indicators(zero_vol)
    assert (s_zero["Vol_Ratio"].dropna() == 1.0).all()
    # Scanner Vol_Ratio は診断列で、学習特徴量（FEATURE_COLUMNS）には含まれない。
    assert "Vol_Ratio" not in scanner.FEATURE_COLUMNS

    # 出来高が変動すれば Scanner Vol_Ratio は 1.0 固定ではなく比になる。
    varying = fixture_mixed()
    varying["Volume"] = np.resize(np.array([100.0, 50.0, 200.0]), len(varying))
    s_var = scanner.add_indicators(varying)
    assert s_var["Vol_Ratio"].dropna().nunique() > 1

    # App Vol_Ratio は Volume の値に一切依存しない（0 でも 100 でも同一）。
    base = fixture_mixed()
    app_base = run_app_load(base)
    app_zero = run_app_load(zero_vol)
    assert np.allclose(
        app_base["Vol_Ratio"].to_numpy(),
        app_zero["Vol_Ratio"].to_numpy(),
        equal_nan=True,
    )


def test_syn_fx_004_missing_volume(monkeypatch):
    """004: Volume 列の欠落。App は任意列として継続できる。
    Scanner add_indicators は Volume を直接参照するため KeyError。
    bars() 経由では欠損 Volume が 0.0 補完される（経路差を分けて記録）。"""
    no_vol = fixture_mixed().drop(columns=["Volume"])

    with pytest.raises(KeyError):
        scanner.add_indicators(no_vol)

    via_bars = run_scanner_bars(monkeypatch, no_vol)
    assert via_bars is not None and len(via_bars) > 0

    app_res = run_app_load(no_vol)
    assert app_res is not None and len(app_res) > 0


def test_syn_fx_005_warmup_boundaries():
    """005: warm-up 境界。各特徴量の有効化タイミングが App/Scanner それぞれの
    定義どおりであることを記録する。"""
    s1 = scanner.add_indicators(fixture_f1_rising())
    s2 = scanner.add_indicators(fixture_f2_falling())
    # Scanner: ATR は Wilder EWM min_periods=14 → 位置 13。
    assert _first_valid_pos(s1["ATR"]) == 13
    # Scanner: RSI は delta 先頭 NaN + min_periods=14 → 位置 14、RSI_Diff は diff(3) → 17。
    assert _first_valid_pos(s2["RSI"]) == 14
    assert _first_valid_pos(s2["RSI_Diff"]) == 17
    # Scanner: ADX は ATR 有効化に連動 → 13。rolling(20) 系は 19、rolling(14) 系は 13。
    assert _first_valid_pos(s1["ADX"]) == 13
    assert _first_valid_pos(s1["BB_PctB"]) == 19
    assert _first_valid_pos(s1["CCI"]) == 19
    assert _first_valid_pos(s1["Dev_SMA20"]) == 19
    assert _first_valid_pos(s1["Stoch_K"]) == 13
    assert _first_valid_pos(s1["Return_5"]) == 5
    assert _first_valid_pos(s1["ROC"]) == 10
    # EMA 系は min_periods 指定がなく初期行から値が出る（200 本 warm-up ではない）。
    assert _first_valid_pos(s1["Dev_EMA200"]) == 0
    assert _first_valid_pos(s1["MACD_Hist_Ratio"]) == 0
    # 長さ境界: 13 本では ATR 有効行 0、14 本で 1 行。
    assert scanner.add_indicators(fixture_f1_rising(13))["ATR"].notna().sum() == 0
    assert scanner.add_indicators(fixture_f1_rising(14))["ATR"].notna().sum() == 1

    # App: 返却 DataFrame の先頭行は、最も遅い Vol_Ratio
    # （ATR rolling 50 / min_periods=10、ATR 自体は位置 6 から有効）のため
    # 元系列の位置 15 になる。
    raw = fixture_f1_rising()
    app_res = run_app_load(raw)
    assert raw.index.get_loc(app_res.index[0]) == 15
    # App は加工後 80 本未満なら None（入力 80 本 → 末尾除外で 79 本 → None）。
    assert run_app_load(fixture_f1_rising(80)) is None
    assert run_app_load(fixture_f1_rising(81)) is not None


def test_syn_fx_006_no_future_leakage():
    """006: signal bar 以降の OHLC だけを変更しても、それ以前の特徴量は不変。
    変化した場合は未来情報混入として失敗する。Target は未来窓依存なので対象外。"""
    full = fixture_mixed()
    t_pos = 150
    ts_t = full.index[t_pos]
    alt = full.copy()
    alt.iloc[t_pos + 1:] = alt.iloc[t_pos + 1:] * 1.05

    s_full = scanner.add_indicators(full)
    s_alt = scanner.add_indicators(alt)
    assert np.allclose(
        s_full.loc[:ts_t, scanner.FEATURE_COLUMNS].to_numpy(dtype=float),
        s_alt.loc[:ts_t, scanner.FEATURE_COLUMNS].to_numpy(dtype=float),
        equal_nan=True,
    )

    # App は末尾足を落とすため、t を残す prefix は t+1 まで含める。
    a_full = run_app_load(full)
    a_prefix = run_app_load(full.iloc[: t_pos + 2])
    a_alt = run_app_load(alt)
    assert ts_t in a_prefix.index and ts_t in a_alt.index
    row_full = a_full.loc[ts_t, APP_FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.allclose(
        row_full, a_prefix.loc[ts_t, APP_FEATURE_COLUMNS].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert np.allclose(
        row_full, a_alt.loc[ts_t, APP_FEATURE_COLUMNS].to_numpy(dtype=float),
        equal_nan=True,
    )


# =========================================================
# 第2群: ラベル境界（SYN-FX-007〜012）
#
# 手法: まず素の合成系列で各実装を実行し、signal bar の ATR・基準価格を
# 実測する（第1パス）。次に未来バーの High/Low を実測 barrier に対して
# 意図した位置に置いた系列を作り、再実行して Target を確認する（第2パス）。
# ATR を固定値で仮定せず実測値から barrier を組み立てるため、丸め誤差で
# 境界がずれることはない。
# =========================================================


def _scanner_feature_frame(raw):
    """bars() 内部と同じ手順で feature_df を再現する（第1パス用）。"""
    return scanner.add_indicators(raw.iloc[:-1]).dropna(
        subset=scanner.FEATURE_COLUMNS + ["ATR", "Close"]
    )


def _engineer_scanner(raw, feat, pos, mode):
    """Scanner 用: 未来バー（特徴量位置 pos+1..pos+9）の High/Low を操作する。"""
    raw = raw.copy()
    entry = feat["Open"].iloc[pos + 1]
    atr = feat["ATR"].iloc[pos]
    for k in range(1, 10):
        ts = feat.index[pos + k]
        if mode == "up" and k == 1:
            raw.at[ts, "High"] = entry + atr + 0.002
            raw.at[ts, "Low"] = entry - 0.5 * atr
        elif mode == "up_exact" and k == 1:
            raw.at[ts, "High"] = entry + atr  # barrier ちょうど（>= で到達）
            raw.at[ts, "Low"] = entry - 0.5 * atr
        elif mode == "down" and k == 1:
            raw.at[ts, "High"] = entry + 0.5 * atr
            raw.at[ts, "Low"] = entry - atr - 0.002
        elif mode == "both" and k == 1:
            raw.at[ts, "High"] = entry + atr + 0.002
            raw.at[ts, "Low"] = entry - atr - 0.002
        elif mode == "eighth" and k == 8:
            raw.at[ts, "High"] = entry + atr + 0.002
            raw.at[ts, "Low"] = entry - 0.5 * atr
        elif mode == "none":
            cap_high = min(raw.at[ts, "High"], entry + 0.5 * atr)
            cap_low = max(raw.at[ts, "Low"], entry - 0.5 * atr)
            raw.at[ts, "High"] = max(raw.at[ts, "Open"], raw.at[ts, "Close"], cap_high)
            raw.at[ts, "Low"] = min(raw.at[ts, "Open"], raw.at[ts, "Close"], cap_low)
    raw["High"] = raw[["High", "Open", "Close"]].max(axis=1)
    raw["Low"] = raw[["Low", "Open", "Close"]].min(axis=1)
    return raw


def _app_barriers(df, pos):
    close = df["Close"].iloc[pos]
    atr = df["ATR"].iloc[pos]
    spread = 0.2 * 0.01  # USDJPY=X: DEFAULT_SPREAD_PIPS 0.2, pip 0.01
    distance = max(atr, 2 * spread) + spread
    return close, distance, close + distance, close - distance


def _engineer_app(raw, df, pos, mode):
    """App 用: 未来バー（加工後位置 pos+1..pos+9）の High/Low を操作する。"""
    raw = raw.copy()
    close, distance, upper, lower = _app_barriers(df, pos)
    for k in range(1, 10):
        ts = df.index[pos + k]
        if mode == "up" and k == 1:
            raw.at[ts, "High"] = upper + 0.001
            raw.at[ts, "Low"] = close - 0.3 * distance
        elif mode == "up_exact" and k == 1:
            raw.at[ts, "High"] = upper  # barrier ちょうど
            raw.at[ts, "Low"] = close - 0.3 * distance
        elif mode == "down" and k == 1:
            raw.at[ts, "High"] = close + 0.3 * distance
            raw.at[ts, "Low"] = lower - 0.001
        elif mode == "both" and k == 1:
            raw.at[ts, "High"] = upper + 0.001
            raw.at[ts, "Low"] = lower - 0.001
        elif mode == "eighth" and k == 8:
            raw.at[ts, "High"] = upper + 0.001
            raw.at[ts, "Low"] = close - 0.3 * distance
        elif mode == "none":
            cap_high = min(raw.at[ts, "High"], close + 0.3 * distance)
            cap_low = max(raw.at[ts, "Low"], close - 0.3 * distance)
            raw.at[ts, "High"] = max(raw.at[ts, "Open"], raw.at[ts, "Close"], cap_high)
            raw.at[ts, "Low"] = min(raw.at[ts, "Open"], raw.at[ts, "Close"], cap_low)
    raw["High"] = raw[["High", "Open", "Close"]].max(axis=1)
    raw["Low"] = raw[["Low", "Open", "Close"]].min(axis=1)
    return raw


def test_syn_fx_007_reference_price_difference(monkeypatch):
    """007: App ラベル用 Close と Scanner の次足 Open に差を作ると、同じ未来
    経路でも参照価格差によりラベルが異なり得る。差をエラーとして隠さず、
    参照基準の違いとして記録する（どちらかを正解として統一しない）。"""
    raw = fixture_mixed()
    feat = _scanner_feature_frame(raw)
    df_a = run_app_load(raw)
    common = [ts for ts in feat.index if ts in df_a.index]
    ts_sig = common[120]
    p_s = feat.index.get_loc(ts_sig)
    close_i = df_a.at[ts_sig, "Close"]
    atr_a = df_a.at[ts_sig, "ATR"]
    atr_s = feat.at[ts_sig, "ATR"]
    app_dist = max(atr_a, 2 * 0.002) + 0.002
    next_ts = feat.index[p_s + 1]

    # ケース1: 次足 Open を signal Close より上へ gap → Scanner の上側 barrier が
    # App より遠くなる。間の価格への到達は App のみ +1。
    mod = raw.copy()
    mod.at[next_ts, "Open"] = close_i + 0.006
    app_upper = close_i + app_dist
    scan_upper = close_i + 0.006 + atr_s
    assert scan_upper - app_upper > 0.002  # 間に置ける幅があること
    high_mid = (app_upper + scan_upper) / 2
    for k in range(1, 9):
        ts = feat.index[p_s + k]
        if k == 1:
            mod.at[ts, "High"] = high_mid
            mod.at[ts, "Low"] = close_i + 0.004
        else:
            mod.at[ts, "High"] = min(mod.at[ts, "High"], close_i + 0.008)
            mod.at[ts, "Low"] = max(mod.at[ts, "Low"], close_i + 0.004)
    mod = normalize_ohlc(mod)
    out_s = run_scanner_bars(monkeypatch, mod)
    out_a = run_app_load(mod)
    assert out_a.at[ts_sig, "Target"] == 1
    assert out_s.at[ts_sig, "Target"] == 0

    # ケース2: 次足 Open を上へ gap → Scanner の下側 barrier が App より
    # 近くなる。間の価格への到達は Scanner のみ -1。
    mod2 = raw.copy()
    mod2.at[next_ts, "Open"] = close_i + 0.006
    scan_lower = close_i + 0.006 - atr_s
    app_lower = close_i - app_dist
    assert scan_lower - app_lower > 0.002
    low_mid = (scan_lower + app_lower) / 2
    for k in range(1, 9):
        ts = feat.index[p_s + k]
        if k == 1:
            mod2.at[ts, "Low"] = low_mid
            mod2.at[ts, "High"] = close_i + 0.004
        else:
            mod2.at[ts, "Low"] = max(mod2.at[ts, "Low"], close_i - 0.002)
            mod2.at[ts, "High"] = min(mod2.at[ts, "High"], close_i + 0.004)
    mod2 = normalize_ohlc(mod2)
    out_s2 = run_scanner_bars(monkeypatch, mod2)
    out_a2 = run_app_load(mod2)
    assert out_a2.at[ts_sig, "Target"] == 0
    assert out_s2.at[ts_sig, "Target"] == -1


def test_syn_fx_008_first_touch(monkeypatch):
    """008: 8本以内に上側だけ/下側だけ到達 → 最初に到達した側のラベル。
    barrier ちょうどの到達（High >= upper / Low <= lower）も含める。"""
    raw = fixture_mixed()
    feat = _scanner_feature_frame(raw)
    pos = 120
    ts_sig = feat.index[pos]
    for mode, expect in (("up", 1), ("down", -1), ("up_exact", 1), ("eighth", 1)):
        out = run_scanner_bars(monkeypatch, _engineer_scanner(raw, feat, pos, mode))
        assert out.at[ts_sig, "Target"] == expect, f"scanner mode={mode}"

    df_a = run_app_load(raw)
    ts_a = df_a.index[pos]
    for mode, expect in (("up", 1), ("down", -1), ("up_exact", 1), ("eighth", 1)):
        out = run_app_load(_engineer_app(raw, df_a, pos, mode))
        assert out.at[ts_a, "Target"] == expect, f"app mode={mode}"


def _pnl_rows(n=16):
    """仮想損益用の行: entry=100.000, ATR=0.010 → upper 100.010 / lower 99.990。"""
    rows = pd.DataFrame(index=_idx(n))
    rows["Future_Open_1"] = 100.0
    rows["Future_Close"] = 100.0
    rows["ATR"] = 0.010
    rows["ADX"] = 20.0
    rows["ATR_Ratio"] = 0.0001
    for step in range(1, 9):
        rows[f"Future_High_{step}"] = 100.005
        rows[f"Future_Low_{step}"] = 99.995
    return rows


def test_syn_fx_009_same_bar_ambiguity(monkeypatch):
    """009: 同一足で上下 barrier 双方に到達。学習ラベルは両実装とも neutral(0)。
    Scanner 仮想損益は学習ラベルと異なり、予測方向に対して損切り側を先に
    評価する別規則であることを確認する（ラベルと損益を混同しない）。
    仮想損益は walk_forward_validation の内包関数を AST 抽出して実行する。"""
    raw = fixture_mixed()
    feat = _scanner_feature_frame(raw)
    pos = 120
    ts_sig = feat.index[pos]
    out = run_scanner_bars(monkeypatch, _engineer_scanner(raw, feat, pos, "both"))
    assert out.at[ts_sig, "Target"] == 0

    df_a = run_app_load(raw)
    ts_a = df_a.index[pos]
    out_a = run_app_load(_engineer_app(raw, df_a, pos, "both"))
    assert out_a.at[ts_a, "Target"] == 0

    rows = _pnl_rows()
    rows.at[rows.index[0], "Future_High_1"] = 100.020  # upper 到達
    rows.at[rows.index[0], "Future_Low_1"] = 99.980   # lower も到達
    for direction in (1, -1):
        preds = np.zeros(len(rows), dtype=int)
        preds[0] = direction
        summary, trades = SIMULATED_PNL(preds, rows, 0.2, 0.01)
        assert summary["trades"] == 1
        assert trades[0]["exit_reason"] == "stop_loss_or_ambiguous_bar"
        # 損切り側で決済: gross -1.0 pip, costs 0.2+0.5+0.2=0.9 → net -1.9
        assert trades[0]["net_pips"] == pytest.approx(-1.9)

    # 対照: 上側のみ到達なら買い方向は take_profit。
    rows_up = _pnl_rows()
    rows_up.at[rows_up.index[0], "Future_High_1"] = 100.020
    preds = np.zeros(len(rows_up), dtype=int)
    preds[0] = 1
    _, trades_up = SIMULATED_PNL(preds, rows_up, 0.2, 0.01)
    assert trades_up[0]["exit_reason"] == "take_profit"
    assert trades_up[0]["net_pips"] == pytest.approx(0.1)


def test_syn_fx_010_timeout_neutral(monkeypatch):
    """010: 8本の未来窓でどちらにも到達しない → Target は 0（timeout/neutral）
    のまま残り、未来窓不足（NaN）とは区別される。"""
    raw = fixture_mixed()
    feat = _scanner_feature_frame(raw)
    pos = 120
    ts_sig = feat.index[pos]
    out = run_scanner_bars(monkeypatch, _engineer_scanner(raw, feat, pos, "none"))
    assert ts_sig in out.index
    assert out.at[ts_sig, "Target"] == 0

    df_a = run_app_load(raw)
    ts_a = df_a.index[pos]
    out_a = run_app_load(_engineer_app(raw, df_a, pos, "none"))
    assert out_a.at[ts_a, "Target"] == 0


def test_syn_fx_011_insufficient_future_window(monkeypatch):
    """011: 未来窓が 8 本に満たない末端サンプルは学習対象に残らない。
    App は末尾 8 行の Target が NaN のまま返り、Scanner は末尾 8 行が
    学習 DataFrame から除外される。"""
    raw = fixture_mixed()
    df_a = run_app_load(raw)
    assert df_a["Target"].iloc[-8:].isna().all()
    assert df_a["Target"].iloc[:-8].notna().all()

    out = run_scanner_bars(monkeypatch, raw)
    feat = out.attrs["latest_features"]
    assert len(out) == len(feat) - scanner.LOOKAHEAD_BARS
    assert out.index[-1] == feat.index[-scanner.LOOKAHEAD_BARS - 1]


def test_syn_fx_012_spread_effect(monkeypatch):
    """012: App の spread 設定だけを変えると barrier/ラベルが変わり得る。
    Scanner の学習ラベルは spread を参照しない（ATR のみ）ことを確認する。
    spread の変更は AST 抽出 namespace 側の定数 dict に対してのみ行い、
    app.py 本体は変更しない。"""
    raw = fixture_mixed()
    df_a = run_app_load(raw)
    pos = 120
    ts_a = df_a.index[pos]
    close, distance, upper, lower = _app_barriers(df_a, pos)

    # 既定 spread(0.2 pip) の barrier をわずかに超える到達を作る。
    mod = raw.copy()
    for k in range(1, 9):
        ts = df_a.index[pos + k]
        if k == 1:
            mod.at[ts, "High"] = upper + 0.0005
            mod.at[ts, "Low"] = close - 0.3 * distance
        else:
            mod.at[ts, "High"] = min(mod.at[ts, "High"], close + 0.3 * distance)
            mod.at[ts, "Low"] = max(mod.at[ts, "Low"], close - 0.3 * distance)
    mod = normalize_ohlc(mod)
    out_default = run_app_load(mod)
    assert out_default.at[ts_a, "Target"] == 1

    # spread を大きくすると barrier が広がり、同じ到達ではラベルが 0 になる。
    original = APP_LOAD_NS["DEFAULT_SPREAD_PIPS"]
    try:
        APP_LOAD_NS["DEFAULT_SPREAD_PIPS"] = {"USDJPY=X": 20.0}
        out_wide = run_app_load(mod)
    finally:
        APP_LOAD_NS["DEFAULT_SPREAD_PIPS"] = original
    assert out_wide.at[ts_a, "Target"] == 0

    # Scanner の学習ラベルは spread の入力自体を持たず、同じ系列では
    # ATR 基準のラベル（この系列では上側到達で +1）になる。
    out_s = run_scanner_bars(monkeypatch, mod)
    assert out_s.at[ts_a, "Target"] == 1



# =========================================================
# 第3群: 分割・時刻・欠損（SYN-FX-013〜019）
# =========================================================


def _wf_frame(n=400):
    """walk_forward_validation 用の合成 DataFrame。

    特徴量は seed 固定の乱数、Target は 3 クラス循環、
    Future_* は barrier 未到達の穏やかな値にする。
    """
    rng = np.random.default_rng(42)
    data = {c: rng.normal(0, 1, n) for c in scanner.FEATURE_COLUMNS}
    df = pd.DataFrame(data, index=_idx(n))
    df["Target"] = np.resize([1, -1, 0], n)
    df["ATR"] = 0.01
    df["Close"] = 100.0
    df["Future_Open_1"] = 100.0
    df["Future_Close"] = 100.0
    df["ADX"] = 20.0
    df["ATR_Ratio"] = 0.0001
    for step in range(1, 9):
        df[f"Future_High_{step}"] = 100.005
        df[f"Future_Low_{step}"] = 99.995
    return df


_WF_CACHE = {}


def _wf_result():
    if "res" not in _WF_CACHE:
        _WF_CACHE["res"] = scanner.walk_forward_validation(
            _wf_frame(), pair_name="SYNTHETIC"
        )
    return _WF_CACHE["res"]


def _app_analyze_frame(n=300, one_class=False):
    rng = np.random.default_rng(7)
    data = {c: rng.normal(0, 1, n) for c in APP_FEATURE_COLUMNS}
    df = pd.DataFrame(data, index=_idx(n))
    df["Target"] = 0 if one_class else np.resize([1, -1, 0], n)
    df["Close"] = 100.0
    df["RSI"] = 50.0
    df["ADX"] = 20.0
    df["ATR_Ratio"] = 0.0001
    return df


def test_syn_fx_013_purge_embargo(monkeypatch, tmp_path):
    """013: train/test 境界の直前・直後でラベル窓 8 本が跨がない。

    App の holdout は train 末尾を 8 行 purge し、Scanner の
    walk-forward も各 fold の train 末尾を 8 行 purge する（purge_gap=8）。
    """
    # --- App: analyze_signal_with_backtest の時系列 holdout ---
    monkeypatch.chdir(tmp_path)  # モデル保存先を repo 外へ
    analyze, fake_st = fresh_app_analyze()
    df = _app_analyze_frame(300)
    analyze(df, None, "USDJPY=X", "1h")
    metrics = fake_st.session_state["model_eval_metrics"]
    # ラベル済み 300 行 -> test_n = min(120, max(40, 20%)) = 60, split=240,
    # train_end = 240 - 8 = 232。train 232 + purge 8 + test 60 = 300。
    assert metrics["test_count"] == 60
    assert metrics["train_count"] == 232
    assert metrics["train_count"] + scanner.LOOKAHEAD_BARS + metrics["test_count"] == 300
    assert metrics["test_start"] == str(df.index[240])

    # --- Scanner: walk_forward_validation の fold 構造 ---
    res = _wf_result()
    details = {d["fold"]: d for d in res["fold_details"]}
    # fold_size = 400 // 4 = 100。fold=2 は train_end=200 -> train 192 行、
    # fold=3 は train_end=300 -> train 292 行。いずれも末尾 8 行が purge される。
    assert details[2]["train_rows"] == 200 - 8
    assert details[3]["train_rows"] == 300 - 8
    assert all(d["purge_gap"] == 8 for d in details.values())


def test_syn_fx_014_walk_forward_fold_boundaries():
    """014: fold 境界が隣接する系列で、各 test fold が時系列順・連続・
    非重複であること。train は過去側のみで、cutoff は 8 行前。
    fold の skip 条件（train 行数不足など）も記録する。
    """
    frame = _wf_frame()
    res = _wf_result()
    details = sorted(res["fold_details"], key=lambda d: d["fold"])
    # fold=1 は train_rows=92 < MIN_TRAIN_ROWS(180) のため skip される（現行仕様）。
    assert [d["fold"] for d in details] == [2, 3]
    assert details[0]["test_start"] == frame.index[200].isoformat()
    assert details[0]["test_end"] == frame.index[299].isoformat()
    assert details[1]["test_start"] == frame.index[300].isoformat()
    assert details[1]["test_end"] == frame.index[399].isoformat()
    assert all(d["test_rows"] == 100 for d in details)
    # test 区間は [200,300) と [300,400) で重複せず、時系列順に隣接する。
    assert details[0]["test_end"] < details[1]["test_start"]


def test_syn_fx_015_one_class_and_zero_trades(monkeypatch, tmp_path):
    """015: テスト側が 1 クラスのみ / 取引 0 件のケース。

    数値を無理に作らず、安全な扱い（失敗 dict・fallback・None 混じりの
    summary）になることを確認する。
    """
    # Scanner: Target が 1 クラスのみ -> 検証は失敗 dict（BA は None）。
    one = _wf_frame()
    one["Target"] = 0
    res = scanner.walk_forward_validation(one, pair_name="SYNTHETIC")
    assert res["passed"] is False
    assert res["reason"] == "目的変数の種類が不足"
    assert res["balanced_accuracy"] is None

    # Scanner 仮想損益: 全予測 neutral（取引 0 件）の summary は
    # trades=0, net_pips=0, win_rate/avg_pips=None, max_drawdown=0。
    rows = _pnl_rows()
    summary, trades = SIMULATED_PNL(np.zeros(len(rows), dtype=int), rows, 0.2, 0.01)
    assert trades == []
    assert summary == {
        "trades": 0,
        "net_pips": 0.0,
        "win_rate_pct": None,
        "avg_pips_per_trade": None,
        "max_drawdown_pips": 0.0,
    }
    # pip_size <= 0 でも同じ安全な summary になる。
    preds = np.zeros(len(rows), dtype=int)
    preds[0] = 1
    summary_p, trades_p = SIMULATED_PNL(preds, rows, 0.2, 0.0)
    assert trades_p == [] and summary_p["trades"] == 0

    # 実経路でも、ペア設定が無い（pip_size=0）fold の仮想損益は 0 件 summary。
    for detail in _wf_result()["fold_details"]:
        assert detail["simulated_pnl"]["trades"] == 0
        assert detail["simulated_pnl"]["win_rate_pct"] is None

    # App: Target が 1 クラスのみ -> fallback（様子見・データ不足）を返し、
    # 評価指標は記録されない（BA を算出しない）。
    monkeypatch.chdir(tmp_path)
    analyze, fake_st = fresh_app_analyze()
    out = analyze(_app_analyze_frame(300, one_class=True), None, "USDJPY=X", "1h")
    assert out[0] == "WAIT (様子見)"
    assert out[4] == "データ不足"
    assert "model_eval_metrics" not in fake_st.session_state



def test_syn_fx_016_invalid_ohlc_inputs(monkeypatch):
    """016: NaN / +Inf / ゼロ価格 / High<Low / 重複・逆順 timestamp を
    1 異常ずつ投入し、各異常が最終特徴量・Target に残るかを記録する。

    現行実装の監査事実として、Stage 1 で定義された不正 OHLC 条件
    （High<Low、High/Low と Open/Close の包含関係、非正価格など）は
    両実装で除外される。一方、本テストで扱う NaN / Inf / 重複・逆順
    timestamp 等の結果は個別ケースの観測であり、あらゆる OHLC 異常を
    包括的に検証したことを意味しない。既存 Stage 1 テストとの重複もある。
    """
    base = fixture_mixed()
    mid_ts = base.index[150]
    outcomes = {}

    # NaN 行: 両実装とも前処理で除外される。
    f = base.copy()
    f.at[mid_ts, "Close"] = np.nan
    outcomes["nan_excluded"] = (
        mid_ts not in run_scanner_bars(monkeypatch, f).index,
        mid_ts not in run_app_load(f).index,
    )
    assert outcomes["nan_excluded"] == (True, True)

    # +Inf 行: 両実装とも学習行には残らない。Scanner の学習特徴量は全て有限。
    f = base.copy()
    f.at[mid_ts, "Close"] = np.inf
    s_inf = run_scanner_bars(monkeypatch, f)
    outcomes["inf_excluded"] = (
        mid_ts not in s_inf.index,
        mid_ts not in run_app_load(f).index,
    )
    assert outcomes["inf_excluded"] == (True, True)
    assert np.isfinite(s_inf[scanner.FEATURE_COLUMNS].to_numpy(dtype=float)).all()

    # Close=0: 両実装とも除外される（比率計算の NaN 化 -> dropna）。
    f = base.copy()
    f.at[mid_ts, "Close"] = 0.0
    outcomes["zero_close_excluded"] = (
        mid_ts not in run_scanner_bars(monkeypatch, f).index,
        mid_ts not in run_app_load(f).index,
    )
    assert outcomes["zero_close_excluded"] == (True, True)

    # High<Low: 現行の段階1入力検証で両実装とも不正行を除外する。
    f = base.copy()
    hi, lo = f.at[mid_ts, "High"], f.at[mid_ts, "Low"]
    f.at[mid_ts, "High"], f.at[mid_ts, "Low"] = lo, hi
    outcomes["high_lt_low_excluded"] = (
        mid_ts not in run_scanner_bars(monkeypatch, f).index,
        mid_ts not in run_app_load(f).index,
    )
    assert outcomes["high_lt_low_excluded"] == (True, True)

    # 重複 timestamp: 現行の段階1入力検証では App / Scanner とも keep=last で一意化。
    f = pd.concat([base, base.iloc[[150]]]).sort_index()
    s_dup = run_scanner_bars(monkeypatch, f)
    a_dup = run_app_load(f)
    assert s_dup.index.is_unique
    assert a_dup.index.is_unique

    # 逆順 timestamp: 両実装とも sort され、昇順の index で返る。
    f = base.iloc[::-1]
    assert run_scanner_bars(monkeypatch, f).index.is_monotonic_increasing
    assert run_app_load(f).index.is_monotonic_increasing

    # timezone-naive 1h: 両実装ともUTCと推定せず fail-closed で停止する。
    f = base.copy()
    f.index = f.index.tz_localize(None)
    s_naive = run_scanner_bars(monkeypatch, f)
    a_naive = run_app_load(f)
    assert s_naive is None
    assert a_naive is None

    print("SYN-FX-016 outcomes:", outcomes)


def test_syn_fx_017_4h_time_boundaries(monkeypatch):
    """017: 4h 時刻境界。

    Scanner と App は、完全な H1 4 本で構成される 4h 区間を採用し、
    途中に欠損を含む不完全区間を除外する。App は最後の完全区間も採用
    する。これらの完全性・最終区間の挙動は既存 Stage 2 テストでも検証
    されており、本テストは別の合成条件で回帰確認する。なお、時刻境界・
    origin・timezone-naive 入力の扱いなど、両実装の差は個別に評価する。
    """
    # --- Scanner: 途中の 4h 区間から H1 を 1 本欠落させると、その区間は不採用 ---
    big = fixture_seeded_walk(1200)  # 300 区間
    pos = 400 - (400 % 4)
    bin_ts = big.index[pos]
    big_gap = big.drop(index=[big.index[pos + 1]])
    out_s = run_scanner_bars(monkeypatch, big_gap, interval="4h")
    assert out_s is not None
    feat_s = out_s.attrs["latest_features"]
    assert bin_ts not in feat_s.index
    assert feat_s.index.tz is not None and str(feat_s.index.tz) == "UTC"

    # --- Scanner: timezone-naive index は 4h 経路で UTC として扱われる ---
    naive = big_gap.copy()
    naive.index = naive.index.tz_localize(None)
    out_n = run_scanner_bars(monkeypatch, naive, interval="4h")
    assert out_n is not None
    assert str(out_n.attrs["latest_features"].index.tz) == "UTC"

    # --- Scanner: 完成した最後の 4h 区間は（過去時刻のため）採用される ---
    big2 = fixture_seeded_walk(1204)  # 301 区間すべて完成
    out_s2 = run_scanner_bars(monkeypatch, big2, interval="4h")
    assert big2.index[1200] in out_s2.attrs["latest_features"].index

    # --- App: 不完全な区間（H1 が 3 本）はScanner同様に除外 ---
    app4 = fixture_seeded_walk(404)  # 101 区間
    apos = 200 - (200 % 4)
    abin_ts = app4.index[apos]
    last_bin_ts = app4.index[400]
    app4_gap = app4.drop(index=[app4.index[apos + 1]])
    out_a = run_app_load(app4_gap, interval="4h")
    assert out_a is not None and len(out_a) > 0
    assert abin_ts not in out_a.index
    assert str(out_a.index.tz) == "UTC"
    # --- App: 最後の完全な4時間区間は採用される ---
    assert last_bin_ts in out_a.index



def test_syn_fx_018_feature_nan_window_compression(monkeypatch):
    """018: Scanner は特徴量 NaN 行を除去してから Target を作るため、
    未来 8 本のうち 1 行が落ちると、未来窓は残存行で圧縮され元の
    9 本目まで延びる。App は Target 作成後に特徴量 dropna を行うため、
    同じ feature-only NaN は Target の 8 本窓から除かれない。
    現行の順序差を再現して記録する（どちらを正解とは決めない）。
    """
    raw = fixture_mixed()
    feat = _scanner_feature_frame(raw)
    pos = 120
    ts_sig = feat.index[pos]
    entry = feat["Open"].iloc[pos + 1]
    atr = feat["ATR"].iloc[pos]
    ts_nan = feat.index[pos + 5]    # 未来窓内の 1 行を特徴量 NaN にする
    ts_touch = feat.index[pos + 9]  # 元データの 9 本目で上側到達させる

    mod = raw.copy()
    for k in range(1, 10):
        ts = feat.index[pos + k]
        if ts == ts_touch:
            mod.at[ts, "High"] = entry + atr + 0.002
            mod.at[ts, "Low"] = entry - 0.5 * atr
        else:
            mod.at[ts, "High"] = min(mod.at[ts, "High"], entry + 0.5 * atr)
            mod.at[ts, "Low"] = max(mod.at[ts, "Low"], entry - 0.5 * atr)

    # 対照（行除去なし）: 9 本目の到達は窓の外 -> Target は 0。
    mod = normalize_ohlc(mod)
    control = run_scanner_bars(monkeypatch, mod)
    assert control.at[ts_sig, "Target"] == 0

    # 隔離 stub: add_indicators の実結果に 1 セルだけ NaN を注入する
    # （本番コードは変更しない。注入はテスト側 wrapper のみ）。
    original_add = scanner.add_indicators

    def stub_add(df):
        out = original_add(df)
        if ts_nan in out.index:
            out.at[ts_nan, "RSI"] = np.nan
        return out

    monkeypatch.setattr(scanner, "add_indicators", stub_add)
    compressed = run_scanner_bars(monkeypatch, mod)
    # 特徴量 NaN による時刻欠損をまたぐ窓は Scanner の連続性判定で除外。
    assert ts_nan not in compressed.attrs["latest_features"].index
    assert ts_sig not in compressed.index
    assert compressed.attrs["label_audit"].loc[ts_sig, "invalid_reason"] == "discontinuous"

    # App 側: 未来窓内の 1 行が特徴量 NaN（ゼロレンジ足 -> ヒゲ比率 NaN）で
    # 出力から落ちても、Target の窓は元の行のまま数えられ、9 本目の
    # 到達は窓外のままになる。
    df_a = run_app_load(raw)
    p_a = 120
    ts_a = df_a.index[p_a]
    close_a, dist_a, upper_a, _lower_a = _app_barriers(df_a, p_a)
    zero_ts = df_a.index[p_a + 2]
    prev_close = float(df_a["Close"].iloc[p_a + 1])
    mod_a = raw.copy()
    for k in range(1, 10):
        ts = df_a.index[p_a + k]
        if ts == zero_ts:
            mod_a.at[ts, "Open"] = prev_close
            mod_a.at[ts, "High"] = prev_close
            mod_a.at[ts, "Low"] = prev_close
            mod_a.at[ts, "Close"] = prev_close
        elif k == 9:
            mod_a.at[ts, "High"] = upper_a + 0.001
            mod_a.at[ts, "Low"] = close_a - 0.3 * dist_a
        else:
            mod_a.at[ts, "High"] = min(mod_a.at[ts, "High"], close_a + 0.3 * dist_a)
            mod_a.at[ts, "Low"] = max(mod_a.at[ts, "Low"], close_a - 0.3 * dist_a)
    mod_a = normalize_ohlc(mod_a)
    out_a = run_app_load(mod_a)
    assert zero_ts not in out_a.index  # 特徴量 NaN 行は出力から落ちる
    assert out_a.at[ts_a, "Target"] == 0  # しかし窓は圧縮されない


def test_syn_fx_019_weekday_gap_vs_forward_policy(monkeypatch):
    """019: 欠損 H1 時刻による 8 本未来窓と forward gap policy の不一致。

    学習ラベル（App/Scanner）は欠損行を落として残存行で 8 本を数えるが、
    Scanner の forward 評価は平日の時間ギャップがあると評価を保留し
    （None）、週末の市場休場ギャップだけを許容する。現行挙動を記録する。
    """
    # --- 学習側（Scanner bars）: 平日3時間欠損を含む未来窓は除外 ---
    start_w = pd.Timestamp("2026-10-26 00:00", tz="UTC")  # Monday
    raw_w = fixture_mixed(290, start=start_w)
    feat_w = _scanner_feature_frame(raw_w)
    pos_w = 120
    ts_sig_w = feat_w.index[pos_w]
    gap_ts = [feat_w.index[pos_w + k] for k in (2, 3, 4)]
    entry_w = feat_w["Open"].iloc[pos_w + 1]
    atr_w = feat_w["ATR"].iloc[pos_w]
    surviving = [
        feat_w.index[pos_w + k]
        for k in range(1, 14)
        if feat_w.index[pos_w + k] not in gap_ts
    ]
    touch8 = surviving[7]  # 残存行で 8 本目（元時刻では 11 時間先）
    mod_w = raw_w.copy()
    for ts in surviving[:8]:
        if ts == touch8:
            mod_w.at[ts, "High"] = entry_w + atr_w + 0.002
            mod_w.at[ts, "Low"] = entry_w - 0.5 * atr_w
        else:
            mod_w.at[ts, "High"] = min(mod_w.at[ts, "High"], entry_w + 0.5 * atr_w)
            mod_w.at[ts, "Low"] = max(mod_w.at[ts, "Low"], entry_w - 0.5 * atr_w)
    mod_w = normalize_ohlc(mod_w.drop(index=gap_ts))
    out_w = run_scanner_bars(monkeypatch, mod_w)
    assert ts_sig_w not in out_w.index
    assert out_w.attrs["label_audit"].loc[ts_sig_w, "invalid_reason"] == "discontinuous"

    # --- forward 評価側: 平日の 3 時間ギャップは評価保留（None） ---
    def fwd_frame(timestamps):
        return pd.DataFrame(
            [(100.05, 100.06, 100.04, 100.05)] * len(timestamps),
            index=pd.DatetimeIndex(timestamps),
            columns=["Open", "High", "Low", "Close"],
        ).astype(float)

    sig_wd = pd.Timestamp("2026-10-28 02:00", tz="UTC")  # Wednesday
    ts_wd = [sig_wd + pd.Timedelta(hours=h) for h in (1, 2, 6, 7, 8, 9, 10, 11)]
    pred = {
        "direction": "買い",
        "reference_price": 100.0,
        "atr": 0.1,
        "pip_size": 0.01,
        "spread_pips": 0.0,
        "signal_bar_time": sig_wd.isoformat(),
    }
    assert scanner.evaluate_forward_prediction(pred, fwd_frame(ts_wd)) is None

    # --- forward 評価側: 週末ギャップ（金曜->月曜、80 時間以内）は許容され評価される ---
    sig_we = pd.Timestamp("2026-10-23 12:00", tz="UTC")  # Friday
    ts_we = [
        pd.Timestamp("2026-10-26 00:00", tz="UTC") + pd.Timedelta(hours=h)
        for h in range(8)
    ]
    pred_we = dict(pred, signal_bar_time=sig_we.isoformat())
    outcome = scanner.evaluate_forward_prediction(pred_we, fwd_frame(ts_we))
    assert outcome is not None and outcome["status"] == "NOT_REACHED"


def test_network_no_attempts_during_suite():
    """suite 全体（自己確認テストを除く）でネットワーク接続試行が
    0 件だったことを、遮断ガードの記録で確認する。
    """
    assert NETWORK_ATTEMPTS == []
    print("NETWORK_ATTEMPTS during suite:", NETWORK_ATTEMPTS)
