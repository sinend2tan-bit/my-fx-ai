"""ネットワークを使わない最小限のスモークテスト。

目的は「構文エラーや import エラーで本番スキャンが起動直後に死ぬ」事故
(例: 2026-10-24 の SyntaxError) を CI で事前に止めること。
scanner.py は `if __name__ == "__main__":` ガードがあるため、
import してもスキャンは実行されない。
"""

import py_compile

import pandas as pd
import pytest

import scanner


def test_all_python_files_compile():
    for path in ("scanner.py", "fx_scanner_simple.py", "app.py"):
        py_compile.compile(path, doraise=True)


def test_safe_float():
    assert scanner.safe_float("1.5") == 1.5
    assert scanner.safe_float(2) == 2.0
    assert scanner.safe_float("not-a-number") is None
    assert scanner.safe_float(None, default=0.0) == 0.0
    assert scanner.safe_float(float("inf")) is None


def test_load_state_missing_file(tmp_path, monkeypatch):
    # 状態ファイルが無い (キャッシュの冷間スタート) でも落ちず、空 dict を返す。
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scanner, "STATE_FILE", str(tmp_path / "scan_state.json"))
    assert scanner.load_state() == {}


def test_load_state_broken_json(tmp_path, monkeypatch):
    state = tmp_path / "scan_state.json"
    state.write_text("{broken json", encoding="utf-8")
    monkeypatch.setattr(scanner, "STATE_FILE", str(state))
    assert scanner.load_state() == {}


def test_flatten_yf_columns_multiindex():
    columns = pd.MultiIndex.from_tuples(
        [("Open", "USDJPY=X"), ("Close", "USDJPY=X")]
    )
    df = pd.DataFrame([[1.0, 2.0]], columns=columns)
    out = scanner.flatten_yf_columns(df.copy())
    assert list(out.columns) == ["Open", "Close"]


def test_contains_news_keyword():
    assert scanner.contains_news_keyword("FOMC holds rates steady") is True
    assert scanner.contains_news_keyword("local weather is sunny") is False



# =========================================================
# フォワード検証（ネットワーク不要・合成データ）
# =========================================================

def _forward_frame(bars_after, signal_close=100.0):
    rows = [(100.0, 100.2, 99.8, signal_close)] + bars_after
    idx = pd.date_range("2026-01-05 00:00+00:00", periods=len(rows), freq="1h")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def _forward_prediction(frame, direction="買い"):
    return {
        "id": scanner.forward_prediction_id("USDJPY=X", frame.index[0].isoformat()),
        "ticker": "USDJPY=X",
        "direction": direction,
        "reference_price": 100.0,
        "atr": 1.0,
        "signal_bar_time": frame.index[0].isoformat(),
        "pip_size": 0.01,
        "spread_pips": 0.2,
        "status": "PENDING",
    }


def _flat_bars(n, close=100.3):
    return [(100.1, 100.6, 99.6, close)] * n


def test_forward_buy_hit():
    frame = _forward_frame(
        [(100.1, 100.4, 99.6, 100.2), (100.2, 101.3, 100.1, 101.0)] + _flat_bars(6)
    )
    out = scanner.evaluate_forward_prediction(_forward_prediction(frame), frame)
    assert out["status"] == "HIT"
    assert out["exit_price"] == 101.0
    assert out["gross_pips"] == 100.0
    # コスト（スプレッド0.2＋スリッページ0.5＋スワップストレス0.2）を控除した純損益
    assert out["net_pips"] == 99.1


def test_forward_buy_miss_and_both_hit_bar():
    frame = _forward_frame([(100.0, 100.3, 98.8, 99.1)] + _flat_bars(7))
    out = scanner.evaluate_forward_prediction(_forward_prediction(frame), frame)
    assert out["status"] == "MISS"
    assert out["net_pips"] == -100.9

    # 同じ足で上下両方に触れたら、足内の順序不明のため不利側（MISS）を先に扱う
    frame = _forward_frame([(100.0, 101.5, 98.5, 100.0)] + _flat_bars(7))
    out = scanner.evaluate_forward_prediction(_forward_prediction(frame), frame)
    assert out["status"] == "MISS"


def test_forward_sell_hit():
    frame = _forward_frame(
        [(99.9, 100.6, 99.5, 99.7), (99.7, 100.2, 98.7, 99.0)] + _flat_bars(6)
    )
    out = scanner.evaluate_forward_prediction(
        _forward_prediction(frame, "売り"), frame
    )
    assert out["status"] == "HIT"
    assert out["gross_pips"] == 100.0


def test_forward_not_reached_and_pending_when_short():
    bars = _flat_bars(8)
    bars[2] = (100.1, 100.6, 99.4, 100.2)
    bars[7] = (100.1, 100.6, 99.6, 100.3)
    frame = _forward_frame(bars)
    out = scanner.evaluate_forward_prediction(_forward_prediction(frame), frame)
    assert out["status"] == "NOT_REACHED"
    assert out["gross_pips"] == 30.0
    assert out["mae_pips"] == 60.0

    # 評価期間の足が8本そろっていなければ評価しない（PENDINGのまま残す）
    frame = _forward_frame(_flat_bars(5))
    assert scanner.evaluate_forward_prediction(_forward_prediction(frame), frame) is None


def test_forward_record_dedupes_and_skips_holds():
    predictions = []
    candidate = {
        "name": "米ドル/円 (USD/JPY)",
        "ticker": "USDJPY=X",
        "decision": "候補あり（手動確認のみ）",
        "direction": "買い",
        "price": 100.0,
        "atr": 1.0,
        "signal_bar_time": "2026-01-05T00:00:00+00:00",
    }
    hold = dict(candidate, decision="検証条件未達・保留")
    assert scanner.record_forward_predictions(predictions, [candidate, hold]) == 1
    # 同じシグナル足の候補は二重に記録しない
    assert scanner.record_forward_predictions(predictions, [candidate]) == 0
    assert len(predictions) == 1
    assert predictions[0]["status"] == "PENDING"


def test_forward_summary_hit_rate_definition():
    stats = scanner.new_forward_stats()
    for status in ("HIT", "MISS", "NOT_REACHED"):
        scanner.update_forward_stats(
            stats, {"direction": "買い"},
            {"status": status, "net_pips": 10.0, "mae_pips": 5.0},
        )
    view = scanner.build_forward_summary(
        stats, [], 3, 0, tracking_since="2026-01-05T00:00:00+09:00"
    )
    # 的中率は HIT / (HIT + MISS)。NOT_REACHED は分母に入れない
    assert view["hit_rate_pct"] == 50.0
    assert view["tracking_since"] == "2026-01-05T00:00:00+09:00"

def _forward_frame_at(times, rows):
    idx = pd.DatetimeIndex(times)
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def test_forward_weekday_gap_blocks_evaluation():
    # 平日の途中に足の欠け（3時間分の抜け）があるウィンドウは評価しない。
    # 欠けた足を詰めて8本そろえても、PENDING のまま残すのが仕様。
    signal = "2026-01-05 00:00+00:00"  # 月曜
    times = [signal,
             "2026-01-05 01:00+00:00", "2026-01-05 02:00+00:00",
             # 03:00〜05:00 が欠けている（02:00 の次が 06:00）
             "2026-01-05 06:00+00:00", "2026-01-05 07:00+00:00",
             "2026-01-05 08:00+00:00", "2026-01-05 09:00+00:00",
             "2026-01-05 10:00+00:00", "2026-01-05 11:00+00:00"]
    rows = [(100.0, 100.2, 99.8, 100.0)] + _flat_bars(8)
    frame = _forward_frame_at(times, rows)
    prediction = _forward_prediction(frame)
    prediction["signal_bar_time"] = signal.replace(" ", "T")
    assert scanner.evaluate_forward_prediction(prediction, frame) is None


def test_forward_weekend_gap_still_evaluates():
    # 金曜→月曜の通常の週末休場（約51時間）は欠損とみなさず、評価する。
    signal = "2026-01-02 20:00+00:00"  # 金曜
    times = [signal,
             "2026-01-02 21:00+00:00",
             "2026-01-05 00:00+00:00", "2026-01-05 01:00+00:00",
             "2026-01-05 02:00+00:00", "2026-01-05 03:00+00:00",
             "2026-01-05 04:00+00:00", "2026-01-05 05:00+00:00",
             "2026-01-05 06:00+00:00"]
    rows = [(100.0, 100.2, 99.8, 100.0)] + _flat_bars(8)
    frame = _forward_frame_at(times, rows)
    prediction = _forward_prediction(frame)
    prediction["signal_bar_time"] = signal.replace(" ", "T")
    out = scanner.evaluate_forward_prediction(prediction, frame)
    assert out is not None
    assert out["status"] == "NOT_REACHED"


def test_forward_trim_keeps_all_pending():
    evaluated = [{"id": f"old{i}", "status": "HIT"} for i in range(4)]
    pending = [{"id": f"pend{i}", "status": "PENDING"} for i in range(4)]
    mixed = evaluated + pending
    kept = scanner.trim_forward_predictions(mixed, max_items=5)
    # 超過分の3件は評価済みの古い順に落ち、PENDING は全て残る
    assert [item["id"] for item in kept] == ["old3", "pend0", "pend1", "pend2", "pend3"]

    # PENDING だけで上限を超える場合は全て保持する（未評価を失わない）
    only_pending = [{"id": f"p{i}", "status": "PENDING"} for i in range(6)]
    assert len(scanner.trim_forward_predictions(only_pending, max_items=5)) == 6

    # 上限以下ならそのまま
    assert scanner.trim_forward_predictions(only_pending[:3], max_items=5) == only_pending[:3]


# =========================================================
# ニュース3系統化・記録と採用の分離（ネットワーク不要・系統スタブ）
# =========================================================

from datetime import timedelta


def _stub_article(title, hours_ago=1.0, publisher="テスト媒体", url=None,
                  source="テスト系統"):
    published = scanner.now_jst() - timedelta(hours=hours_ago)
    if url is None:
        url = f"https://example.com/news/{sum(map(ord, title)) % 10007}"
    return {
        "title": title,
        "publisher": publisher,
        "url": url,
        "published_at": published.isoformat(),
        "_published": published,
        "source": source,
    }


def _stub_source(articles=None, errors=None, ok=True):
    def _fetch(ticker):
        return {
            "articles": articles or [],
            "errors": errors or [],
            "ok": ok,
            "method": "stub",
        }
    return _fetch


def _stub_all_sources(monkeypatch, google, central, yahoo):
    monkeypatch.setattr(scanner, "fetch_google_news_articles", google)
    monkeypatch.setattr(scanner, "fetch_central_bank_articles", central)
    monkeypatch.setattr(scanner, "fetch_yahoo_articles", yahoo)


# キーワードを含まない見出し（test_contains_news_keyword で確認済みの文面）
_NEUTRAL_TITLE = "local weather is sunny"
# 重要キーワードを含む見出し（同上）
_RISK_TITLE = "FOMC holds rates steady"


def test_news_all_sources_fail_is_error(monkeypatch):
    failing = _stub_source(errors=["stub: ConnectionError: down"], ok=False)
    _stub_all_sources(monkeypatch, failing, failing, failing)
    news = scanner.fetch_news("USDJPY=X")
    # 全系統の失敗は「ニュースなし」ではなく error として区別される
    assert news["news_status"] == "error"
    assert news["news_retrieval_status"] == "error"
    assert len(news["news_sources"]) == 3
    assert all(entry["error"] for entry in news["news_sources"])


def test_news_fetcher_exception_is_isolated(monkeypatch):
    # 取得関数自体が例外を出しても、他の系統で判定でき、全体は止まらない
    def boom(ticker):
        raise RuntimeError("kaboom")

    ok_source = _stub_source(articles=[_stub_article(_NEUTRAL_TITLE)])
    _stub_all_sources(monkeypatch, boom, ok_source, _stub_source())
    news = scanner.fetch_news("USDJPY=X")
    assert news["news_status"] == "recent_articles_no_risk_keywords"
    assert news["news_article_count"] == 1
    google_entry = news["news_sources"][0]
    assert google_entry["error"]


def test_news_all_sources_empty(monkeypatch):
    empty = _stub_source()
    _stub_all_sources(monkeypatch, empty, empty, empty)
    news = scanner.fetch_news("USDJPY=X")
    # 取得は成功して記事が0件なら error ではなく empty
    assert news["news_status"] == "empty"
    assert news["news_retrieval_status"] == "no_articles_returned"


def test_news_partial_failure_uses_remaining_sources(monkeypatch):
    failing = _stub_source(errors=["stub: TimeoutError"], ok=False)
    central = _stub_source(articles=[_stub_article(_NEUTRAL_TITLE)])
    _stub_all_sources(monkeypatch, failing, central, _stub_source())
    news = scanner.fetch_news("USDJPY=X")
    assert news["news_status"] == "recent_articles_no_risk_keywords"
    assert news["news_recent_count"] == 1
    by_name = {entry["source"]: entry for entry in news["news_sources"]}
    assert by_name["Googleニュース"]["error"]
    assert by_name["中央銀行フィード"]["error"] is None
    assert by_name["中央銀行フィード"]["article_count"] == 1
    assert by_name["中央銀行フィード"]["latest_published_jst"]


def test_news_old_articles_only(monkeypatch):
    old = _stub_source(articles=[
        _stub_article(_NEUTRAL_TITLE, hours_ago=48.0),
        _stub_article("another old story", hours_ago=30.0),
    ])
    _stub_all_sources(monkeypatch, old, _stub_source(), _stub_source())
    news = scanner.fetch_news("USDJPY=X")
    assert news["news_status"] == "no_recent_articles"
    assert news["news_article_count"] == 2
    assert news["news_recent_count"] == 0


def test_news_dedupes_same_story_across_sources(monkeypatch):
    # 同じ記事が媒体違い・リンク違いで来ても、タイトル正規化で1件になる
    google = _stub_source(articles=[_stub_article(
        "静かな市場 - テスト媒体", publisher="テスト媒体",
        url="https://example.com/a",
    )])
    yahoo = _stub_source(articles=[_stub_article(
        "静かな市場", publisher="別媒体", url="https://example.com/b",
    )])
    _stub_all_sources(monkeypatch, google, _stub_source(), yahoo)
    news = scanner.fetch_news("USDJPY=X")
    assert news["news_article_count"] == 1
    assert len(news["news_headlines"]) == 1


def test_news_risk_keyword_status(monkeypatch):
    risky = _stub_source(articles=[_stub_article(_RISK_TITLE)])
    _stub_all_sources(monkeypatch, risky, _stub_source(), _stub_source())
    news = scanner.fetch_news("USDJPY=X")
    assert news["news_status"] == "recent_articles_with_risk_keywords"
    assert news["news_risk_detected"] is True
    allowed, _reason = scanner.news_allows_candidate(news)
    assert allowed is False


def test_central_bank_feed_parses_rss2(monkeypatch):
    rss2 = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0"><channel>'
        "<item><title>Bank Rate maintained at 3.75%</title>"
        "<link>https://example.com/boe/1</link>"
        "<pubDate>Mon, 05 Jan 2026 00:00:00 GMT</pubDate></item>"
        "<item><title>Office opening hours notice</title>"
        "<link>https://example.com/boe/2</link>"
        "<pubDate>Mon, 05 Jan 2026 00:00:00 GMT</pubDate></item>"
        "</channel></rss>"
    ).encode("utf-8")
    monkeypatch.setattr(scanner, "_fetch_url_bytes", lambda url: rss2)
    articles, error = scanner._fetch_central_bank_feed(
        "英イングランド銀行（公式）", "https://example.com/rss"
    )
    assert error is None
    # 金融政策関連の見出しだけが残り、日時も解釈できる
    assert len(articles) == 1
    assert articles[0]["title"].startswith("Bank Rate")
    assert articles[0]["_published"] is not None


def test_central_bank_feed_parses_rss1_dc_date(monkeypatch):
    rss1 = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns="http://purl.org/rss/1.0/" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<item rdf:about="https://example.com/rba/1">'
        "<title>Monetary Policy Decision: cash rate held</title>"
        "<link>https://example.com/rba/1</link>"
        "<dc:date>2026-01-05T00:00:00Z</dc:date>"
        "</item></rdf:RDF>"
    ).encode("utf-8")
    monkeypatch.setattr(scanner, "_fetch_url_bytes", lambda url: rss1)
    articles, error = scanner._fetch_central_bank_feed(
        "豪準備銀行（公式）", "https://example.com/rss"
    )
    assert error is None
    assert len(articles) == 1
    assert articles[0]["_published"] is not None


def test_central_bank_feed_invalid_response(monkeypatch):
    # 不正な応答（XMLでない）でも例外を出さず、エラーとして返す
    monkeypatch.setattr(scanner, "_fetch_url_bytes", lambda url: b"not xml")
    articles, error = scanner._fetch_central_bank_feed(
        "テスト銀行", "https://example.com/bad"
    )
    assert articles == []
    assert error


def test_google_news_feed_parses_items(monkeypatch):
    rss = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0"><channel>'
        "<item><title>ドル円は小動き - テスト通信</title>"
        "<link>https://example.com/gn/1</link>"
        "<pubDate>Mon, 05 Jan 2026 00:00:00 GMT</pubDate>"
        "<source>テスト通信</source></item>"
        "</channel></rss>"
    ).encode("utf-8")
    monkeypatch.setattr(scanner, "_fetch_url_bytes", lambda url: rss)
    result = scanner.fetch_google_news_articles("USDJPY=X")
    assert result["ok"] is True
    assert result["errors"] == []
    assert len(result["articles"]) == 1
    assert result["articles"][0]["publisher"] == "テスト通信"
    assert result["articles"][0]["source"] == "Googleニュース"


# =========================================================
# 記録と採用の分離（analyze_pair のスタブ検証）
# =========================================================

def _news_result(status, risk=False):
    return {
        "news_checked_at_jst": "2026-01-05T00:00:00+09:00",
        "news_status": status,
        "news_retrieval_status": "stub",
        "news_article_count": 1,
        "news_recent_count": 1,
        "news_risk_detected": risk,
        "news_risk_reason": "",
        "news_risk_headlines": [],
        "news_headlines": [],
        "news_source": "stub",
        "news_retrieval_method": "stub",
        "news_retrieval_errors": [],
        "news_sources": [],
    }


def _stub_pair_pipeline(monkeypatch, news, direction="買い", trend="中立"):
    latest_index = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-05T00:00:00+09:00")]
    )
    latest_data = pd.DataFrame(
        {"Close": [150.0], "ATR": [1.0]}, index=latest_index
    )

    def fake_bars(ticker, interval="1h", period="180d"):
        frame = pd.DataFrame({"Close": [150.0]}, index=latest_index)
        frame.attrs["latest_data"] = latest_data
        frame.attrs["latest_history"] = frame
        return frame

    monkeypatch.setattr(scanner, "fetch_news", lambda ticker: dict(news))
    monkeypatch.setattr(scanner, "bars", fake_bars)
    monkeypatch.setattr(scanner, "timeframe_trend", lambda frame: trend)
    monkeypatch.setattr(
        scanner, "validate_latest_bar", lambda data: (True, "ok")
    )
    monkeypatch.setattr(scanner, "data_age_minutes", lambda frame: 10.0)
    monkeypatch.setattr(
        scanner, "fast_move_detected", lambda history: (False, 0.0)
    )
    monkeypatch.setattr(
        scanner,
        "walk_forward_validation",
        lambda frame, pair_name="": {
            "passed": True,
            "reason": "ok",
            "directional_pnl": {
                "buy": {"passed": True},
                "sell": {"passed": True},
            },
        },
    )
    monkeypatch.setattr(
        scanner,
        "analyze",
        lambda frame, validation: {
            "direction": direction,
            "buy_pct": 60.0,
            "sell_pct": 20.0,
            "neutral_pct": 20.0,
            "reason": "stub",
        },
    )


_PAIR_CONFIG = {"ticker": "USDJPY=X", "spread_pips": 0.2, "pip": 0.01}
_PAIR_NAME = "米ドル/円 (USD/JPY)"


def test_news_risk_holds_candidate_but_recordable(monkeypatch):
    # 重要キーワード検出: 採用は従来どおり保留。ただし記録対象にはなる
    news = _news_result("recent_articles_with_risk_keywords", risk=True)
    _stub_pair_pipeline(monkeypatch, news)
    result = scanner.analyze_pair(_PAIR_NAME, _PAIR_CONFIG)
    assert result["decision"] == "ニュース確認・判定保留"
    assert result["forward_recordable"] is True


def test_news_unavailable_holds_candidate_but_recordable(monkeypatch):
    # ニュース取得失敗でも、他の条件を満たせば記録対象になる
    news = _news_result("error")
    _stub_pair_pipeline(monkeypatch, news)
    result = scanner.analyze_pair(_PAIR_NAME, _PAIR_CONFIG)
    assert result["decision"] == "ニュース確認・判定保留"
    assert result["forward_recordable"] is True


def test_news_ok_makes_candidate_and_recordable(monkeypatch):
    news = _news_result("recent_articles_no_risk_keywords")
    _stub_pair_pipeline(monkeypatch, news)
    result = scanner.analyze_pair(_PAIR_NAME, _PAIR_CONFIG)
    assert result["decision"] == "候補あり（手動確認のみ）"
    assert result["forward_recordable"] is True


def test_trend_conflict_not_recordable_and_news_priority(monkeypatch):
    # 上位足と食い違う場合は記録対象にもならない。
    # ニュース保留と競合しても、判定の優先順位は従来どおりニュースが先
    news = _news_result("error")
    _stub_pair_pipeline(monkeypatch, news, direction="買い", trend="下降")
    result = scanner.analyze_pair(_PAIR_NAME, _PAIR_CONFIG)
    assert result["decision"] == "ニュース確認・判定保留"
    assert result["forward_recordable"] is False


def test_trend_conflict_decision_after_news_ok(monkeypatch):
    news = _news_result("recent_articles_no_risk_keywords")
    _stub_pair_pipeline(monkeypatch, news, direction="買い", trend="下降")
    result = scanner.analyze_pair(_PAIR_NAME, _PAIR_CONFIG)
    assert result["decision"] == "上位足と不一致・保留"
    assert result["forward_recordable"] is False


# =========================================================
# フォワード検証: ニュース未確認の記録と集計分離
# =========================================================

def _pair_result_for_record(**overrides):
    base = {
        "name": _PAIR_NAME,
        "ticker": "USDJPY=X",
        "decision": "ニュース確認・判定保留",
        "direction": "買い",
        "forward_recordable": True,
        "news_status": "error",
        "price": 150.0,
        "atr": 1.0,
        "signal_bar_time": "2026-01-05T00:00:00+09:00",
    }
    base.update(overrides)
    return base


def test_forward_record_news_unverified_and_dedupe():
    predictions = []
    unverified = _pair_result_for_record()
    details = {}
    recorded = scanner.record_forward_predictions(
        predictions, [unverified], details=details
    )
    assert recorded == 1
    assert predictions[0]["news_verified"] is False
    assert predictions[0]["news_status_at_record"] == "error"
    assert details == {"recorded_verified": 0, "recorded_unverified": 1}

    # 記録対象が広がっても、同じシグナル足はIDで二重登録を防ぐ
    assert scanner.record_forward_predictions(predictions, [unverified]) == 0
    assert len(predictions) == 1


def test_forward_record_news_verified_for_candidate():
    predictions = []
    verified = _pair_result_for_record(
        decision="候補あり（手動確認のみ）",
        news_status="recent_articles_no_risk_keywords",
    )
    details = {}
    assert scanner.record_forward_predictions(
        predictions, [verified], details=details
    ) == 1
    assert predictions[0]["news_verified"] is True
    assert details == {"recorded_verified": 1, "recorded_unverified": 0}


def test_forward_record_skips_non_recordable_hold():
    predictions = []
    hold = _pair_result_for_record(
        decision="検証条件未達・保留", forward_recordable=False
    )
    assert scanner.record_forward_predictions(predictions, [hold]) == 0
    assert predictions == []


def _evaluated_record(status, news_verified="missing", net=0.0, mae=0.0):
    record = {
        "id": f"rec-{status}-{news_verified}-{net}",
        "ticker": "USDJPY=X",
        "direction": "買い",
        "status": status,
        "net_pips": net,
        "mae_pips": mae,
    }
    if news_verified != "missing":
        record["news_verified"] = news_verified
    return record


def test_forward_stats_buckets_split():
    records = [
        _evaluated_record("HIT", True, net=10.0, mae=5.0),
        _evaluated_record("MISS", False, net=-20.0, mae=8.0),
        _evaluated_record("NOT_REACHED", "missing", net=3.0, mae=2.0),
        {"id": "pend", "status": "PENDING"},  # 未評価は集計に入らない
    ]
    buckets = scanner.compute_forward_stats_by_news_verified(records)
    assert buckets["verified"]["evaluated_total"] == 1
    assert buckets["verified"]["hit"] == 1
    assert buckets["verified"]["net_pips_total"] == 10.0
    assert buckets["unverified"]["miss"] == 1
    assert buckets["unverified"]["net_pips_total"] == -20.0
    # 旧形式（フィールドなし）は未確認ではなく「不明」区分に入る
    assert buckets["unknown"]["not_reached"] == 1
    assert buckets["unknown"]["evaluated_total"] == 1


def test_forward_stats_init_from_history_keeps_old_records():
    old_record = _evaluated_record("HIT", "missing", net=7.0, mae=1.0)
    previous_state = {
        "predictions": [old_record],
        "forward_stats": {
            "evaluated_total": 1,
            "hit": 1,
            "miss": 0,
            "not_reached": 0,
            "net_pips_total": 7.0,
            "mae_pips_total": 1.0,
        },
    }
    predictions, stats, evaluated_now = (
        scanner.process_forward_verification(previous_state)
    )
    assert evaluated_now == 0
    # 旧レコード自体は書き換えられない（フィールドを後付けしない）
    assert "news_verified" not in predictions[0]
    # 初回は履歴から再計算され、旧形式は「不明」区分に入る
    assert stats["by_news_verified"]["unknown"]["hit"] == 1
    assert stats["by_news_verified"]["unverified"]["evaluated_total"] == 0
    # 全体の累積カウンタは従来どおり保持される
    assert stats["evaluated_total"] == 1
    assert stats["net_pips_total"] == 7.0


def test_forward_update_stats_updates_matching_bucket():
    stats = scanner.new_forward_stats()
    scanner.update_forward_stats(
        stats, {"direction": "買い", "news_verified": True},
        {"status": "HIT", "net_pips": 10.0, "mae_pips": 5.0},
    )
    scanner.update_forward_stats(
        stats, {"direction": "売り", "news_verified": False},
        {"status": "MISS", "net_pips": -9.1, "mae_pips": 3.0},
    )
    scanner.update_forward_stats(
        stats, {"direction": "買い"},  # 旧形式（フィールドなし）
        {"status": "HIT", "net_pips": 1.0, "mae_pips": 1.0},
    )
    assert stats["evaluated_total"] == 3
    assert stats["by_news_verified"]["verified"]["hit"] == 1
    assert stats["by_news_verified"]["unverified"]["miss"] == 1
    assert stats["by_news_verified"]["unknown"]["hit"] == 1

    view = scanner.build_forward_summary(stats, [], 3, 0)
    breakdown = view["news_verified_breakdown"]
    assert breakdown["verified"]["hit_rate_pct"] == 100.0
    assert breakdown["unverified"]["hit_rate_pct"] == 0.0
    assert breakdown["unknown"]["evaluated_total"] == 1
    # 全体の的中率は従来定義のまま（HIT 2 / (HIT 2 + MISS 1)）
    assert view["hit_rate_pct"] == 66.7


def test_step_summary_shows_source_tally_and_forward_breakdown(
    tmp_path, monkeypatch
):
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    pair_results = [{
        "name": _PAIR_NAME,
        "decision": "ニュース確認・判定保留",
        "direction": "買い",
        "buy_pct": 60.0,
        "sell_pct": 20.0,
        "data_freshness_status": "fresh",
        "news_status": "error",
        "news_sources": [
            {"source": "Googleニュース", "article_count": 0,
             "latest_published_jst": None, "error": "Googleニュース: URLError"},
            {"source": "中央銀行フィード", "article_count": 2,
             "latest_published_jst": "2026-01-05T00:00:00+09:00",
             "error": None},
        ],
    }]
    forward_summary = scanner.build_forward_summary(
        scanner.new_forward_stats(), [], 0, 1,
        recorded_verified_now=0, recorded_unverified_now=1,
    )
    scanner.write_step_summary(
        "2026-01-05T00:00:00+09:00", pair_results, ["line"],
        {"error": 1}, {"pairs_total": 1}, forward_summary,
    )
    text = summary_file.read_text(encoding="utf-8")
    assert "取得系統別" in text
    assert "ニュース確認済み 0件・未確認 1件" in text
    assert "評価済み累計（全体）" in text
    assert "確認状態不明（旧形式）" in text


# =========================================================
# 中央銀行フィードの部分成功／重複統合時の情報補完
# =========================================================

_BOJ_RSS_OK = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<rss version="2.0"><channel>'
    "<item><title>金融政策の実施について</title>"
    "<link>https://www.boj.or.jp/example</link>"
    "<pubDate>Mon, 27 Oct 2026 01:00:00 GMT</pubDate></item>"
    "</channel></rss>"
).encode("utf-8")

_FED_RSS_OK = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<rss version="2.0"><channel>'
    "<item><title>Federal Reserve monetary policy statement</title>"
    "<link>https://www.federalreserve.gov/example</link>"
    "<pubDate>Mon, 27 Oct 2026 01:00:00 GMT</pubDate></item>"
    "</channel></rss>"
).encode("utf-8")


def test_central_bank_partial_success(monkeypatch):
    # FRBフィードだけ不正な応答でも、系統は「部分成功」になり、
    # 成功した日銀フィードの記事は使える
    def fake_fetch(url):
        if "boj.or.jp" in url:
            return _BOJ_RSS_OK
        return b"not xml"

    monkeypatch.setattr(scanner, "_fetch_url_bytes", fake_fetch)
    fetched = scanner.fetch_central_bank_articles("USDJPY=X")
    assert fetched["ok"] is True
    assert fetched["status"] == "partial"
    assert fetched["feeds_ok"] == ["日本銀行（公式）"]
    assert fetched["feeds_failed"] == ["米連邦準備制度（公式）"]
    assert fetched["errors"]
    assert "米連邦準備制度（公式）" in fetched["errors"][0]
    assert len(fetched["articles"]) == 1


def test_central_bank_all_feeds_ok_status(monkeypatch):
    def fake_fetch(url):
        return _BOJ_RSS_OK if "boj.or.jp" in url else _FED_RSS_OK

    monkeypatch.setattr(scanner, "_fetch_url_bytes", fake_fetch)
    fetched = scanner.fetch_central_bank_articles("USDJPY=X")
    assert fetched["ok"] is True
    assert fetched["status"] == "ok"
    assert fetched["feeds_failed"] == []
    assert len(fetched["articles"]) == 2


def test_central_bank_all_feeds_fail_status(monkeypatch):
    def fake_fetch(url):
        raise ConnectionError("down")

    monkeypatch.setattr(scanner, "_fetch_url_bytes", fake_fetch)
    fetched = scanner.fetch_central_bank_articles("USDJPY=X")
    assert fetched["ok"] is False
    assert fetched["status"] == "error"
    assert fetched["articles"] == []
    assert len(fetched["feeds_failed"]) == 2
    assert fetched["feeds_ok"] == []


def test_news_sources_show_central_bank_partial(monkeypatch):
    # fetch_news の系統別記録に、中央銀行フィードの部分成功と
    # 成功・失敗したフィード名が残る
    def central(ticker):
        return {
            "articles": [_stub_article(_NEUTRAL_TITLE)],
            "errors": ["米連邦準備制度（公式）: ParseError: broken"],
            "ok": True,
            "method": "中央銀行フィード",
            "status": "partial",
            "feeds_ok": ["日本銀行（公式）"],
            "feeds_failed": ["米連邦準備制度（公式）"],
        }

    _stub_all_sources(monkeypatch, _stub_source(), central, _stub_source())
    news = scanner.fetch_news("USDJPY=X")
    by_name = {entry["source"]: entry for entry in news["news_sources"]}
    entry = by_name["中央銀行フィード"]
    assert entry["status"] == "partial"
    assert entry["feeds_ok"] == ["日本銀行（公式）"]
    assert entry["feeds_failed"] == ["米連邦準備制度（公式）"]
    # 部分成功でも他系統の記事で通常どおり判定できる
    assert news["news_status"] == "recent_articles_no_risk_keywords"


def test_step_summary_shows_partial_source(tmp_path, monkeypatch):
    # サマリーで部分成功が単なる「失敗」に数えられない
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    pair_results = [{
        "name": _PAIR_NAME,
        "decision": "ニュース確認・判定保留",
        "direction": "買い",
        "buy_pct": 60.0,
        "sell_pct": 20.0,
        "data_freshness_status": "fresh",
        "news_status": "recent_articles_no_risk_keywords",
        "news_sources": [
            {"source": "中央銀行フィード", "article_count": 1,
             "latest_published_jst": "2026-01-05T00:00:00+09:00",
             "error": "米連邦準備制度（公式）: ParseError: broken",
             "status": "partial",
             "feeds_ok": ["日本銀行（公式）"],
             "feeds_failed": ["米連邦準備制度（公式）"]},
        ],
    }]
    forward_summary = scanner.build_forward_summary(
        scanner.new_forward_stats(), [], 0, 1,
    )
    scanner.write_step_summary(
        "2026-01-05T00:00:00+09:00", pair_results, ["line"],
        {}, {"pairs_total": 1}, forward_summary,
    )
    text = summary_file.read_text(encoding="utf-8")
    assert "中央銀行フィード 成功0・部分成功1・失敗0" in text


def test_dedupe_backfills_missing_published_and_link():
    recent = scanner.now_jst() - timedelta(hours=1)
    first = {
        "title": "同じ見出し", "publisher": "媒体A", "url": "",
        "published_at": None, "_published": None, "source": "系統1",
    }
    dupe = {
        "title": "同じ見出し - 媒体B", "publisher": "媒体B",
        "url": "https://example.com/story",
        "published_at": recent.isoformat(), "_published": recent,
        "source": "系統2",
    }
    merged = scanner.dedupe_news_articles([first, dupe])
    assert len(merged) == 1
    assert merged[0]["_published"] == recent
    assert merged[0]["published_at"] == recent.isoformat()
    assert merged[0]["url"] == "https://example.com/story"


def test_dedupe_does_not_overwrite_valid_fields():
    # 先行記事に有効な値がある項目は、重複記事の値で上書きしない
    older = scanner.now_jst() - timedelta(hours=2)
    newer = scanner.now_jst() - timedelta(hours=1)
    first = {
        "title": "同じ見出し", "publisher": "媒体A",
        "url": "https://example.com/first",
        "published_at": older.isoformat(), "_published": older,
        "source": "系統1",
    }
    dupe = {
        "title": "同じ見出し", "publisher": "媒体B",
        "url": "https://example.com/dupe",
        "published_at": newer.isoformat(), "_published": newer,
        "source": "系統2",
    }
    merged = scanner.dedupe_news_articles([first, dupe])
    assert len(merged) == 1
    assert merged[0]["_published"] == older
    assert merged[0]["url"] == "https://example.com/first"
    assert merged[0]["publisher"] == "媒体A"


def test_news_backfilled_datetime_counts_as_recent(monkeypatch):
    # 日時不明の先行記事に、重複記事の有効な日時が補完されると、
    # その記事は直近記事として数えられる（日時不明のまま切り捨てない）
    undated = {
        "title": _NEUTRAL_TITLE, "publisher": "媒体A", "url": "",
        "published_at": None, "_published": None, "source": "テスト系統",
    }
    dated = _stub_article(
        _NEUTRAL_TITLE, hours_ago=1.0, publisher="媒体B",
        url="https://example.com/story",
    )
    google = _stub_source(articles=[undated])
    yahoo = _stub_source(articles=[dated])
    _stub_all_sources(monkeypatch, google, _stub_source(), yahoo)
    news = scanner.fetch_news("USDJPY=X")
    assert news["news_article_count"] == 1
    assert news["news_recent_count"] == 1
    assert news["news_status"] == "recent_articles_no_risk_keywords"
    assert news["news_headlines"][0]["published_at"] is not None
    assert news["news_headlines"][0]["url"] == "https://example.com/story"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))



# =========================================================
# アプリのバックアップ束（app.py から対象関数だけ抽出して検証）
# app.py は import すると Streamlit アプリが走るため、AST で関数を取り出す。
# =========================================================

def _extract_app_backup_functions(tmp_path):
    import ast
    import json as json_module
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    # テストの実行場所（chdir 後など）に依存しないよう、このファイルの位置から app.py を引く。
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"load_pair_scan_outcomes", "build_data_backup_bundle", "restore_data_backup_bundle"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    namespace = {
        "json": json_module,
        "datetime": datetime,
        "JST": timezone(timedelta(hours=9)),
        "_json_default": str,
        "DEFAULT_SETTINGS": {"webhook_url": "", "rr_ratio": 1.5},
        "SETTINGS_FILE": str(tmp_path / "user_settings.json"),
        "FORECAST_TRACK_FILE": str(tmp_path / "forecast_outcome_history.json"),
        "ENTRY_TRACK_FILE": str(tmp_path / "entry_candidate_history.json"),
        "TRADE_JOURNAL_FILE": str(tmp_path / "actual_trade_journal.json"),
        "FEEDBACK_FILE": str(tmp_path / "trade_feedback.json"),
        "PAIR_SCAN_OUTCOME_FILE": str(tmp_path / "pair_scan_outcomes.json"),
        "load_settings": lambda: {"webhook_url": "https://discord.com/api/webhooks/dummy", "rr_ratio": 1.5},
        "load_forecast_history": lambda: [{"kind": "forecast"}],
        "load_entry_tracking": lambda: [{"kind": "entry"}],
        "load_trade_journal": lambda: [{"kind": "journal"}],
        "load_feedback_data": lambda: [{"kind": "feedback"}],
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
    return namespace


def test_app_backup_bundle_includes_pair_scan_outcomes(tmp_path, monkeypatch):
    import json as json_module
    monkeypatch.chdir(tmp_path)
    rows = [{"イベントID": "a|1時間足|買い|t1"}, {"イベントID": "b|1時間足|売り|t2"}]
    (tmp_path / "pair_scan_outcomes.json").write_text(json_module.dumps(rows, ensure_ascii=False), encoding="utf-8")
    ns = _extract_app_backup_functions(tmp_path)
    bundle = ns["build_data_backup_bundle"]()
    assert bundle["pair_scan_outcomes"] == rows
    # Webhook URL はバックアップに含めない仕様どおり空になる。
    assert bundle["settings"]["webhook_url"] == ""


def test_app_restore_writes_pair_scan_outcomes(tmp_path, monkeypatch):
    import json as json_module
    monkeypatch.chdir(tmp_path)
    ns = _extract_app_backup_functions(tmp_path)
    rows = [{"イベントID": "x|1時間足|買い|t9"}]
    bundle = {
        "format": "ai_fx_analyzer_backup",
        "settings": {"rr_ratio": 2.0},
        "forecast_history": [], "entry_tracking": [], "trade_journal": [], "feedback_history": [],
        "pair_scan_outcomes": rows,
    }
    ns["restore_data_backup_bundle"](bundle)
    saved = json_module.loads((tmp_path / "pair_scan_outcomes.json").read_text(encoding="utf-8"))
    assert saved == rows


def test_app_restore_legacy_bundle_keeps_pair_scan_outcomes(tmp_path, monkeypatch):
    # pair_scan_outcomes を含まない旧形式のバックアップを復元しても、
    # 現在の pair_scan_outcomes.json を空で上書きしない。
    import json as json_module
    monkeypatch.chdir(tmp_path)
    ns = _extract_app_backup_functions(tmp_path)
    current = [{"イベントID": "keep-me"}]
    (tmp_path / "pair_scan_outcomes.json").write_text(json_module.dumps(current, ensure_ascii=False), encoding="utf-8")
    legacy_bundle = {
        "format": "ai_fx_analyzer_backup",
        "settings": {"rr_ratio": 2.0},
        "forecast_history": [], "entry_tracking": [], "trade_journal": [], "feedback_history": [],
    }
    ns["restore_data_backup_bundle"](legacy_bundle)
    saved = json_module.loads((tmp_path / "pair_scan_outcomes.json").read_text(encoding="utf-8"))
    assert saved == current
