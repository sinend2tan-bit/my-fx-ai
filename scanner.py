import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight


# =========================================================
# 基本設定
# =========================================================

JST = ZoneInfo("Asia/Tokyo")

STATE_FILE = "scan_state.json"
LOOKAHEAD_BARS = 8
# 特徴量計算に必要な最小の足本数（加工後）。最も遅い指標である EMA200 が
# 有効になるには 200 本が必要で、残りは他の指標とラベル付けの余裕分。
# bars() の事前チェックも、この値を基準に経路ごとに算出する。
MIN_FEATURE_BARS = 220
MIN_TRAIN_ROWS = 180
N_WALK_FORWARD_FOLDS = 3
FAST_MOVE_ATR_MULTIPLE = 2.5

# 最新の確定1時間足がこの時間を超えて古い場合は、検証計算を行わず保留する。
MAX_DATA_AGE_MINUTES = 180

# 方向別の仮想損益を確認する最低件数。件数不足は「良好」とみなさず保留する。
MIN_DIRECTIONAL_PNL_TRADES = 10

# ニュースを新しいものとみなす時間
NEWS_MAX_AGE_HOURS = 12

# 重要ニュースを検出した場合は新しい売買候補を保留
NEWS_HOLD_ON_IMPORTANT = True

# ニュースが確認できない場合も新しい売買候補を保留
NEWS_HOLD_IF_UNAVAILABLE = True

# フォワード検証で保持する予測レコードの上限。
# 古い評価済みレコードは切り詰めるが、累積集計（forward_stats）は別に保持するため
# 切り詰めで累積成績が失われることはない。
FORWARD_PREDICTIONS_MAX = 200

MODEL_VERSION = 19

# 仮想検証用の追加コスト（実際の松井証券FXの約定・スワップ値ではありません）。
# スリッページは往復合計の保守的な仮定。スワップは保有時間・方向・日々の値で変動するため、
# この定数は比較用の暫定ストレス値です。実際の取引条件を確認してから調整してください。
SIMULATED_ROUND_TRIP_SLIPPAGE_PIPS = 0.5
SIMULATED_SWAP_STRESS_COST_PIPS = 0.2


PAIRS = {
    "米ドル/円 (USD/JPY)": {
        "ticker": "USDJPY=X",
        "spread_pips": 0.2,
        "pip": 0.01,
    },
    "英ポンド/円 (GBP/JPY)": {
        "ticker": "GBPJPY=X",
        "spread_pips": 1.0,
        "pip": 0.01,
    },
    "ユーロ/円 (EUR/JPY)": {
        "ticker": "EURJPY=X",
        "spread_pips": 0.5,
        "pip": 0.01,
    },
    "豪ドル/円 (AUD/JPY)": {
        "ticker": "AUDJPY=X",
        "spread_pips": 0.6,
        "pip": 0.01,
    },
    "ユーロ/米ドル (EUR/USD)": {
        "ticker": "EURUSD=X",
        "spread_pips": 0.4,
        "pip": 0.0001,
    },
}

# 売買方向を決めるための特徴量
FEATURE_COLUMNS = [
    "Return_1",
    "Return_5",
    "Dev_SMA20",
    "Dev_EMA20",
    "Dev_EMA200",
    "RSI",
    "RSI_Diff",
    "MACD_Hist_Ratio",
    "BB_PctB",
    "ADX",
    "ATR_Ratio",
    "Upper_Wick_Ratio",
    "Lower_Wick_Ratio",
    "Stoch_K",
    "CCI",
    "ROC",
]

# ニュースの重要性を検出するためのキーワード
NEWS_KEYWORDS = [
    # 米国の金融政策
    "federal reserve",
    "fed ",
    " fomc",
    "interest rate",
    "rate decision",
    "rate cut",
    "rate hike",
    "powell",
    "treasury yield",
    "米連邦準備制度",
    "米連邦準備理事会",
    "米金融政策",
    "米政策金利",
    "パウエル",
    "米国債利回り",

    # 日本の金融政策
    "bank of japan",
    " boj ",
    "ueda",
    "日銀",
    "日本銀行",
    "植田総裁",
    "金融政策決定会合",
    "政策金利",

    # 経済指標
    "cpi",
    "consumer price",
    "inflation",
    "ppi",
    "nonfarm payroll",
    "payrolls",
    "employment report",
    "unemployment",
    "jobless claims",
    "gdp",
    "pce price",
    "小売売上高",
    "消費者物価",
    "生産者物価",
    "インフレ",
    "雇用統計",
    "失業率",
    "国内総生産",
    "経済指標",

    # 為替介入・急変動要因
    "intervention",
    "currency intervention",
    "foreign exchange intervention",
    "tariff",
    "trade war",
    "sanctions",
    "geopolitical",
    "military conflict",
    "emergency",
    "介入",
    "為替介入",
    "関税",
    "貿易摩擦",
    "制裁",
    "地政学",
    "軍事衝突",
    "緊急",
]


# =========================================================
# 共通処理
# =========================================================

def now_jst():
    return datetime.now(JST)


def iso_now():
    return now_jst().isoformat()


def safe_float(value, default=None):
    try:
        result = float(value)
        if np.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def timestamp_diagnostic(value):
    """時刻を診断用に日本時間へ変換。naive時刻は勝手にUTC扱いせず明示する。"""
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return f"timezone_unknown:{ts.isoformat()}"
        return ts.tz_convert("Asia/Tokyo").isoformat()
    except Exception:
        return "unparseable"


def news_health_summary(pair_results):
    """ニュース取得・記事有無・新着確認を混同せずに集計する。"""
    statuses = [item.get("news_status", "unknown") for item in pair_results]
    return {
        "retrieval_error_pairs": sum(status == "error" for status in statuses),
        "empty_result_pairs": sum(status == "empty" for status in statuses),
        "articles_returned_but_none_recent_pairs": sum(status == "no_recent_articles" for status in statuses),
        "recent_articles_checked_pairs": sum(status in {
            "recent_articles_no_risk_keywords", "recent_articles_with_risk_keywords"
        } for status in statuses),
        "important_keyword_pairs": sum(status == "recent_articles_with_risk_keywords" for status in statuses),
        "unknown_status_pairs": sum(status == "unknown" for status in statuses),
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "note": "emptyは取得エラーと同義ではありません。記事なし・新着なし・取得エラーを区別しています。",
    }


def compare_previous_run(previous_state, pair_results):
    """直前実行との変化を短く記録する。過去の値を現在の予測に混ぜない。"""
    previous_pairs = previous_state.get("pairs", []) if isinstance(previous_state, dict) else []
    if not isinstance(previous_pairs, list):
        previous_pairs = []
    previous_by_ticker = {
        item.get("ticker"): item for item in previous_pairs
        if isinstance(item, dict) and item.get("ticker")
    }
    comparisons = []
    for item in pair_results:
        old = previous_by_ticker.get(item.get("ticker"))
        if not old:
            comparisons.append({"ticker": item.get("ticker"), "name": item.get("name"), "status": "no_previous_pair_record"})
            continue
        old_age = safe_float(old.get("data_age_minutes"))
        new_age = safe_float(item.get("data_age_minutes"))
        age_delta = round(new_age - old_age, 1) if old_age is not None and new_age is not None else None
        comparisons.append({
            "ticker": item.get("ticker"),
            "name": item.get("name"),
            "status": "compared",
            "previous_decision": old.get("decision"),
            "current_decision": item.get("decision"),
            "decision_changed": old.get("decision") != item.get("decision"),
            "previous_freshness": old.get("data_freshness_status"),
            "current_freshness": item.get("data_freshness_status"),
            "previous_data_age_minutes": old_age,
            "current_data_age_minutes": new_age,
            "data_age_delta_minutes": age_delta,
            "previous_news_status": old.get("news_status"),
            "current_news_status": item.get("news_status"),
        })
    return comparisons


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        print("[STATE LOAD WARNING] scan_state.json is not a JSON object; starting with empty state")
        return {}
    except Exception as exc:
        # 壊れた履歴があってもスキャン全体を止めない。原因はログに残す。
        print(f"[STATE LOAD WARNING] {type(exc).__name__}: {str(exc)[:180]}; starting with empty state")
        return {}


def save_state(state):
    """一時ファイルへ書いてから置換し、途中終了でJSONが壊れるリスクを下げる。"""
    target = os.path.abspath(STATE_FILE)
    target_dir = os.path.dirname(target) or "."
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_dir,
            prefix=".scan_state_",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = f.name
            json.dump(
                state,
                f,
                ensure_ascii=False,
                indent=2,
                default=json_default,
            )
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, target)
        print(f"[STATE SAVE] path={STATE_FILE}, atomic_replace=True")
    except Exception:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


# =========================================================
# ニュース取得・確認
# =========================================================


def parse_news_time(article):
    """
    Yahoo Financeのニュース日時を複数の形式から読み取る。
    数値形式、文字列形式、providerPublishTimeに対応する。
    """
    content = article.get("content", article)

    pub_date = content.get("pubDate")
    if pub_date is not None:
        try:
            if isinstance(
                pub_date,
                (int, float, np.integer, np.floating),
            ):
                return datetime.fromtimestamp(
                    float(pub_date), tz=JST
                )

            if (
                isinstance(pub_date, str)
                and pub_date.strip().isdigit()
            ):
                return datetime.fromtimestamp(
                    float(pub_date.strip()), tz=JST
                )

            timestamp = pd.Timestamp(pub_date)

            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")

            return timestamp.tz_convert(
                "Asia/Tokyo"
            ).to_pydatetime()

        except Exception:
            pass

    for obj in (content, article):
        epoch = obj.get("providerPublishTime")

        if epoch is not None:
            try:
                return datetime.fromtimestamp(
                    float(epoch), tz=JST
                )
            except (TypeError, ValueError, OverflowError, OSError):
                pass

    return None



def extract_news_fields(article):
    """
    Yahoo Financeのニュース形式の違いに対応する。
    """
    content = article.get("content", article)

    title = (
        content.get("title")
        or article.get("title")
        or ""
    )

    provider = content.get("provider")
    if isinstance(provider, dict):
        provider = provider.get("displayName", "")

    if not provider:
        provider = article.get("publisher", "")

    canonical_url = content.get("canonicalUrl")
    if isinstance(canonical_url, dict):
        link = canonical_url.get("url", "")
    else:
        link = canonical_url or content.get("link") or article.get("link") or ""

    published = parse_news_time(article)

    return {
        "title": str(title).strip(),
        "publisher": str(provider or ""),
        "url": str(link or ""),
        "published_at": published.isoformat() if published else None,
        "_published": published,
    }


def contains_news_keyword(title):
    text = " " + str(title).lower() + " "
    return any(keyword.lower() in text for keyword in NEWS_KEYWORDS)



# ニュース取得は3系統（GoogleニュースRSS・中央銀行公式フィード・
# Yahoo Finance）。各系統は個別に失敗を隔離し、1系統の障害・
# タイムアウト・不正な応答で他系統やスキャン全体を止めない。
# RSS の取得は標準ライブラリ urllib のみを使い、新規依存を増やさない。
NEWS_HTTP_TIMEOUT_SECONDS = 8

# ペア別キーワード（Googleニュース検索用）。app.py の見出し取得と同じ。
GOOGLE_NEWS_QUERIES = {
    "USDJPY=X": "ドル円",
    "GBPJPY=X": "ポンド円",
    "EURJPY=X": "ユーロ円",
    "AUDJPY=X": "豪ドル円",
    "EURUSD=X": "ユーロドル",
}

# 中央銀行の公式フィード（通貨コード→フィード）。app.py と同じ集合。
# 公式発表のみで、経済指標カレンダーや速報ニュースの代替ではない。
CENTRAL_BANK_FEEDS = [
    ("JPY", "日本銀行（公式）", "https://www.boj.or.jp/rss/whatsnew.xml"),
    ("USD", "米連邦準備制度（公式）", "https://www.federalreserve.gov/feeds/press_monetary.xml"),
    ("EUR", "欧州中央銀行（公式）", "https://www.ecb.europa.eu/rss/press.html"),
    ("GBP", "英イングランド銀行（公式）", "https://www.bankofengland.co.uk/rss/news"),
    ("AUD", "豪準備銀行（公式）", "https://www.rba.gov.au/rss/rss-cb-media-releases.xml"),
]

# 公式フィードの見出しを絞るマクロ・金融政策関連の語句（app.py と同じ）。
_OFFICIAL_FEED_TERMS = (
    "金融政策", "政策金利", "物価", "経済", "為替", "日銀", "総裁", "金融",
    "monetary", "fomc", "interest rate", "inflation", "economic",
    "federal funds", "policy rate", "bank rate", "cash rate",
)

_RSS10_NS = "http://purl.org/rss/1.0/"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_DC_NS = "http://purl.org/dc/elements/1.1/"


def _fetch_url_bytes(url, timeout=NEWS_HTTP_TIMEOUT_SECONDS):
    """URL を取得して bytes を返す（urllib のみ）。失敗時は例外を送出する。"""
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 FX-analysis-tool"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _parse_feed_datetime(raw):
    """RSS/Atom の日時文字列を JST の datetime へ変換する。読めなければ None。"""
    if not raw:
        return None
    try:
        parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        if parsed is None or pd.isna(parsed):
            return None
        return parsed.tz_convert(JST).to_pydatetime()
    except Exception:
        return None


def _feed_node_text(node, names):
    """候補要素名のうち、最初に見つかった空でないテキストを返す（名前空間対応）。"""
    for name in names:
        element = node.find(name)
        if element is not None and element.text and element.text.strip():
            return element.text.strip()
    return ""


def _make_news_article(title, publisher, url, published, source):
    """系統間で共通の記事形式（extract_news_fields と同じ形）に正規化する。"""
    return {
        "title": str(title).strip(),
        "publisher": str(publisher or ""),
        "url": str(url or ""),
        "published_at": published.isoformat() if published else None,
        "_published": published,
        "source": source,
    }


def fetch_google_news_articles(ticker):
    """Googleニュース検索RSS系統。ペア別キーワード＋直近3日で取得する。

    戻り値: {"articles": [...], "errors": [...], "ok": bool, "method": str}
    失敗しても例外を外に出さず、errors と ok=False で知らせる。
    """
    source_name = "Googleニュース"
    code = str(ticker or "").upper()
    query = GOOGLE_NEWS_QUERIES.get(code)
    if query is None:
        query = code.replace("=X", "") or "為替"
    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote_plus(f"{query} when:3d")
        + "&hl=ja&gl=JP&ceid=JP:ja"
    )
    try:
        root = ET.fromstring(_fetch_url_bytes(url))
    except Exception as exc:
        return {
            "articles": [],
            "errors": [
                f"{source_name}: {type(exc).__name__}: {str(exc)[:120]}"
            ],
            "ok": False,
            "method": None,
        }

    articles = []
    for node in root.findall(".//item")[:30]:
        title = _feed_node_text(node, ["title"])
        if not title:
            continue
        publisher = _feed_node_text(node, ["source"]) or source_name
        link = _feed_node_text(node, ["link"])
        published = _parse_feed_datetime(
            _feed_node_text(node, ["pubDate"])
        )
        articles.append(
            _make_news_article(title, publisher, link, published, source_name)
        )
    return {"articles": articles, "errors": [], "ok": True, "method": source_name}


def _central_bank_feeds_for_ticker(ticker):
    """ペアの2通貨に対応する中央銀行フィードだけを返す。"""
    code = str(ticker or "").upper()
    return [
        (publisher, url)
        for currency, publisher, url in CENTRAL_BANK_FEEDS
        if currency in code
    ]


def _fetch_central_bank_feed(publisher, url):
    """中央銀行フィード1件を取得し、(articles, error) を返す。

    RSS 2.0 / Atom / RSS 1.0（RDF・dc:date）に対応する。見出しは
    マクロ・金融政策関連の語句を含むものだけに絞る（app.py と同じ）。
    失敗しても例外を外に出さない。
    """
    try:
        root = ET.fromstring(_fetch_url_bytes(url))
    except Exception as exc:
        return [], f"{publisher}: {type(exc).__name__}: {str(exc)[:120]}"

    nodes = root.findall(".//item")
    if not nodes:
        nodes = root.findall(f".//{{{_ATOM_NS}}}entry")
    if not nodes:
        nodes = root.findall(f".//{{{_RSS10_NS}}}item")

    articles = []
    for node in nodes:
        title = _feed_node_text(
            node,
            ["title", f"{{{_ATOM_NS}}}title", f"{{{_RSS10_NS}}}title"],
        )
        if not title:
            continue
        if not any(term in title.lower() for term in _OFFICIAL_FEED_TERMS):
            continue
        link = _feed_node_text(node, ["link", f"{{{_RSS10_NS}}}link"])
        if not link:
            for name in ("link", f"{{{_ATOM_NS}}}link"):
                element = node.find(name)
                if element is not None and element.attrib.get("href"):
                    link = element.attrib["href"]
                    break
        raw_date = _feed_node_text(
            node,
            [
                "pubDate",
                f"{{{_ATOM_NS}}}updated",
                f"{{{_ATOM_NS}}}published",
                f"{{{_DC_NS}}}date",
            ],
        )
        published = _parse_feed_datetime(raw_date)
        articles.append(
            _make_news_article(title, publisher, link, published, publisher)
        )
    return articles, None


def fetch_central_bank_articles(ticker):
    """中央銀行公式フィード系統。フィード単位で失敗を隔離する。

    1つでもフィードを取得できれば系統としては ok とし、失敗した
    フィードは errors に記録する。系統の状態はフィード単位の
    成否で区別する：全フィード成功なら ok、一部だけ成功なら
    partial（部分成功）、全フィード失敗なら error。部分成功を
    系統全体の失敗と表示しないための区別である。
    """
    articles = []
    errors = []
    feeds_ok = []
    feeds_failed = []
    for publisher, url in _central_bank_feeds_for_ticker(ticker):
        feed_articles, error = _fetch_central_bank_feed(publisher, url)
        if error:
            errors.append(error)
            feeds_failed.append(publisher)
        else:
            feeds_ok.append(publisher)
            articles.extend(feed_articles)
    if feeds_failed and feeds_ok:
        status = "partial"
    elif feeds_failed:
        status = "error"
    else:
        status = "ok"
    return {
        "articles": articles,
        "errors": errors,
        "ok": bool(feeds_ok),
        "method": "中央銀行フィード",
        "status": status,
        "feeds_ok": feeds_ok,
        "feeds_failed": feeds_failed,
    }


def fetch_yahoo_articles(ticker):
    """Yahoo Finance系統。yf.Search → Ticker.get_news/news の順に試す。"""
    raw_news = []
    errors = []
    method = None
    completed = False

    # 方法1：yf.Search が利用可能なら試す
    search_api = getattr(yf, "Search", None)

    if callable(search_api):
        try:
            search_result = search_api(
                ticker, news_count=10
            )
            completed = True
            candidate = getattr(
                search_result, "news", None
            )

            if isinstance(candidate, list) and candidate:
                raw_news = candidate
                method = "yf.Search"

        except Exception as exc:
            errors.append(
                "yf.Search: "
                + type(exc).__name__
                + ": "
                + str(exc)[:120]
            )

    # 方法2：Ticker.get_news、または従来のTicker.news
    if not raw_news:
        try:
            ticker_obj = yf.Ticker(ticker)
            get_news = getattr(
                ticker_obj, "get_news", None
            )

            if callable(get_news):
                try:
                    candidate = get_news(count=10)
                except TypeError:
                    candidate = get_news()

                method_used = "Ticker.get_news"
            else:
                candidate = ticker_obj.news
                method_used = "Ticker.news"

            completed = True
            if isinstance(candidate, list) and candidate:
                raw_news = candidate
                method = method_used

        except Exception as exc:
            errors.append(
                "Ticker news: "
                + type(exc).__name__
                + ": "
                + str(exc)[:120]
            )

    articles = []
    for article in raw_news:
        if not isinstance(article, dict):
            continue

        try:
            item = extract_news_fields(article)
        except Exception as exc:
            print(
                f"[NEWS PARSE ERROR] "
                f"ticker={ticker}, "
                f"error={type(exc).__name__}"
            )
            continue

        if not item["title"]:
            continue

        item["source"] = "Yahoo Finance"
        articles.append(item)

    return {
        "articles": articles,
        "errors": errors,
        "ok": completed,
        "method": method,
    }


def _news_title_key(title, publisher):
    """重複判定用のタイトル正規化（app.py の _news_title_key と同じ）。

    Googleニュース等のタイトル末尾にある「 - 媒体名」を除去してから
    比較し、同じ記事が媒体違いで重複しないようにする。
    """
    title = str(title or "").strip()
    publisher = str(publisher or "").strip()
    if publisher:
        suffix = f" - {publisher}"
        if title.lower().endswith(suffix.lower()):
            title = title[: -len(suffix)].strip()
    return title.lower()


def _backfill_news_article(kept, duplicate):
    """重複記事から、先行記事に欠けた情報だけを補完する。

    先行記事に有効な値がある項目は上書きしない。特に公開日時が
    不明（None）の先行記事へ重複記事の有効な日時を取り込むことで、
    新しい記事が「日時不明・古い記事」として誤って扱われるのを防ぐ。
    """
    if kept.get("_published") is None and duplicate.get("_published") is not None:
        kept["_published"] = duplicate["_published"]
        kept["published_at"] = (
            duplicate.get("published_at")
            or duplicate["_published"].isoformat()
        )
    # リンクは url / link どちらのキーでも欠けていれば補う。
    if not str(kept.get("url") or kept.get("link") or "").strip():
        link_value = str(
            duplicate.get("url") or duplicate.get("link") or ""
        ).strip()
        if link_value:
            if "url" in kept or "link" not in kept:
                kept["url"] = link_value
            else:
                kept["link"] = link_value
    for key in ("publisher", "source"):
        if not str(kept.get(key) or "").strip() and str(
            duplicate.get(key) or ""
        ).strip():
            kept[key] = duplicate[key]


def dedupe_news_articles(articles):
    """タイトル正規化またはリンクが一致する記事を1件に統合する。

    重複と判定された記事は単に捨てず、先行記事に欠けた情報
    （公開日時・リンク・媒体名など）を重複記事の有効な値で補完する。
    重複判定キーの仕様自体は変えない。
    """
    merged = []
    by_title = {}
    by_link = {}
    for item in articles:
        if not isinstance(item, dict):
            continue
        title_key = _news_title_key(item.get("title"), item.get("publisher"))
        link_key = str(item.get("url") or item.get("link") or "").strip()
        kept = None
        if title_key and title_key in by_title:
            kept = by_title[title_key]
        elif link_key and link_key in by_link:
            kept = by_link[link_key]
        if kept is not None:
            _backfill_news_article(kept, item)
            # 重複記事側のキーも先行記事に寄せ、3件目以降の重複も
            # 同じ先行記事へ統合できるようにする。
            if title_key and title_key not in by_title:
                by_title[title_key] = kept
            if link_key and link_key not in by_link:
                by_link[link_key] = kept
            continue
        if title_key:
            by_title[title_key] = item
        if link_key:
            by_link[link_key] = item
        merged.append(item)
    return merged


# 取得系統は (表示名, 取得関数の名前)。関数名で引くことで、テストから
# 系統ごとに差し替え（スタブ化）できるようにしている。
NEWS_SOURCE_FETCHERS = (
    ("Googleニュース", "fetch_google_news_articles"),
    ("中央銀行フィード", "fetch_central_bank_articles"),
    ("Yahoo Finance", "fetch_yahoo_articles"),
)


def fetch_news(ticker):
    """
    3系統（GoogleニュースRSS・中央銀行公式フィード・Yahoo Finance）から
    ニュースを取得する。各系統の失敗は隔離し、取得失敗を
    「重要ニュースなし」と誤認しない。
    """
    result = {
        "news_checked_at_jst": iso_now(),
        "news_status": "unknown",
        "news_retrieval_status": "unknown",
        "news_article_count": 0,
        "news_recent_count": 0,
        "news_risk_detected": False,
        "news_risk_reason": "",
        "news_risk_headlines": [],
        "news_headlines": [],
        "news_source": "",
        "news_retrieval_method": None,
        "news_retrieval_errors": [],
        "news_sources": [],
    }

    # 系統ごとに取得する。取得関数自体が例外を出しても、ここで隔離して
    # その系統だけを失敗として記録し、残りの系統と処理全体は続ける。
    source_results = []
    for source_name, func_name in NEWS_SOURCE_FETCHERS:
        fetcher = globals().get(func_name)
        try:
            fetched = fetcher(ticker)
            if not isinstance(fetched, dict):
                raise TypeError("invalid fetcher result")
            source_ok = bool(fetched.get("ok"))
            source_results.append({
                "source": source_name,
                "articles": fetched.get("articles") or [],
                "errors": [
                    str(error) for error in (fetched.get("errors") or [])
                ],
                "ok": source_ok,
                "method": fetched.get("method"),
                # 系統が細分状態（中央銀行フィードの partial 等）を返す
                # 場合はそれを使い、なければ ok から単純に決める。
                "status": fetched.get("status")
                or ("ok" if source_ok else "error"),
                "feeds_ok": fetched.get("feeds_ok"),
                "feeds_failed": fetched.get("feeds_failed"),
            })
        except Exception as exc:
            source_results.append({
                "source": source_name,
                "articles": [],
                "errors": [
                    f"{source_name}: {type(exc).__name__}: {str(exc)[:120]}"
                ],
                "ok": False,
                "method": None,
                "status": "error",
                "feeds_ok": None,
                "feeds_failed": None,
            })

    retrieval_errors = []
    for entry in source_results:
        retrieval_errors.extend(entry["errors"])
    result["news_retrieval_errors"] = retrieval_errors[:6]

    # 系統別の取得状況。取得件数は重複排除前の、その系統が返した件数。
    for entry in source_results:
        published_values = [
            article.get("_published")
            for article in entry["articles"]
            if isinstance(article, dict)
            and article.get("_published") is not None
        ]
        source_entry = {
            "source": entry["source"],
            "article_count": len(entry["articles"]),
            "latest_published_jst": (
                max(published_values).isoformat()
                if published_values else None
            ),
            "error": "; ".join(entry["errors"]) if entry["errors"] else None,
            # 系統の状態（ok / partial / error）。中央銀行フィードの
            # 部分成功を系統全体の失敗と表示しないために持つ。
            "status": entry.get("status")
            or ("ok" if entry["ok"] else "error"),
        }
        # 中央銀行フィードはフィード単位の成否も残す。
        if entry.get("feeds_ok") is not None:
            source_entry["feeds_ok"] = entry["feeds_ok"]
        if entry.get("feeds_failed") is not None:
            source_entry["feeds_failed"] = entry["feeds_failed"]
        result["news_sources"].append(source_entry)

    merged = dedupe_news_articles([
        article
        for entry in source_results
        for article in entry["articles"]
    ])
    # 見出しの並びを新しい順（日時不明は末尾）に安定させる。
    merged.sort(
        key=lambda article: (
            article.get("_published") is None,
            -(
                article["_published"].timestamp()
                if article.get("_published") is not None else 0.0
            ),
        )
    )

    contributing = [
        entry["source"] for entry in source_results if entry["articles"]
    ]
    result["news_source"] = " / ".join(contributing)
    methods = [
        entry["method"] or entry["source"]
        for entry in source_results
        if entry["articles"]
    ]
    result["news_retrieval_method"] = (
        ", ".join(methods) if methods else None
    )

    if not merged:
        # 全系統が失敗した場合だけ error とする。一部の系統が成功して
        # 記事が0件だった場合は、取得失敗ではなく empty として区別する。
        any_source_ok = any(entry["ok"] for entry in source_results)
        result["news_status"] = (
            "empty" if any_source_ok else "error"
        )
        result["news_retrieval_status"] = (
            "no_articles_returned" if any_source_ok else "error"
        )

        if any_source_ok:
            result["news_risk_reason"] = (
                "ニュース記事が返されませんでした。"
                "重要ニュースがないとは判断できません。"
            )
        else:
            result["news_risk_reason"] = (
                "複数のニュース取得方法でエラーが発生しました。"
                "重要ニュースがないとは判断できません。"
            )

        print(
            f"[NEWS FETCH] ticker={ticker}, "
            f"status={result['news_status']}, "
            f"method=None, "
            f"errors={len(retrieval_errors)}"
        )
        return result

    result["news_retrieval_status"] = "articles_returned"
    result["news_article_count"] = len(merged)

    current_time = now_jst()
    cutoff = current_time - timedelta(
        hours=NEWS_MAX_AGE_HOURS
    )

    recent_articles = []

    for article in merged:
        item = dict(article)

        if not item.get("title"):
            continue

        published = item.pop("_published", None)
        result["news_headlines"].append(item)

        # 日時が確認できない記事は直近扱いにしない
        if published is not None:
            if published.tzinfo is None:
                published = published.replace(tzinfo=JST)

            if cutoff <= published <= (
                current_time + timedelta(minutes=10)
            ):
                recent_articles.append(item)

    result["news_headlines"] = (
        result["news_headlines"][:10]
    )
    result["news_recent_count"] = len(recent_articles)

    if not recent_articles:
        result["news_status"] = "no_recent_articles"
        result["news_risk_reason"] = (
            f"取得記事の中に過去{NEWS_MAX_AGE_HOURS}時間以内と"
            "確認できる記事がありません。"
        )

        print(
            f"[NEWS FETCH] ticker={ticker}, "
            f"status=no_recent_articles, "
            f"articles={result['news_article_count']}, "
            f"method={result['news_retrieval_method']}"
        )
        return result

    risk_headlines = [
        item for item in recent_articles
        if contains_news_keyword(item["title"])
    ]

    if risk_headlines:
        result["news_status"] = (
            "recent_articles_with_risk_keywords"
        )
        result["news_risk_detected"] = True
        result["news_risk_headlines"] = risk_headlines[:5]
        result["news_risk_reason"] = (
            "直近のニュース見出しに重要キーワードがあります。"
            "影響方向を見出しだけでは判断できないため保留します。"
        )
    else:
        result["news_status"] = (
            "recent_articles_no_risk_keywords"
        )
        result["news_risk_reason"] = (
            "直近記事を取得しましたが、設定した重要キーワードは"
            "検出されませんでした。重要ニュースをすべて網羅する"
            "保証はありません。"
        )

    print(
        f"[NEWS FETCH] ticker={ticker}, "
        f"status={result['news_status']}, "
        f"articles={result['news_article_count']}, "
        f"recent={result['news_recent_count']}, "
        f"method={result['news_retrieval_method']}, "
        f"risk={result['news_risk_detected']}"
    )

    return result


def news_allows_candidate(news):
    """
    ニュース確認の状態に応じて、新規売買候補を許可するか判定する。
    """
    if NEWS_HOLD_ON_IMPORTANT and news.get("news_risk_detected"):
        return False, "重要ニュース検出のため保留"

    usable_statuses = {
        "recent_articles_no_risk_keywords",
        "recent_articles_with_risk_keywords",
    }

    if NEWS_HOLD_IF_UNAVAILABLE and news.get("news_status") not in usable_statuses:
        return False, "ニュースを十分に確認できないため保留"

    return True, ""


# =========================================================
# テクニカル指標
# =========================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], np.nan)


def calculate_atr(df, period=14):
    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()


def add_indicators(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_price = df["Open"]
    volume = df["Volume"]

    df["Return_1"] = close.pct_change()
    df["Return_5"] = close.pct_change(5)

    sma20 = close.rolling(20).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    df["Dev_SMA20"] = close / sma20 - 1
    df["Dev_EMA20"] = close / ema20 - 1
    df["Dev_EMA200"] = close / ema200 - 1

    # FXデータでは出来高が常に0の場合があるため、その場合は中立値を使用
    volume_mean = volume.rolling(20, min_periods=1).mean()

    if volume.isna().all() or (volume.fillna(0) == 0).all():
        df["Vol_Ratio"] = 1.0
    else:
        df["Vol_Ratio"] = (
            volume / volume_mean.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    df["RSI"] = calculate_rsi(close)
    df["RSI_Diff"] = df["RSI"].diff(3)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal

    atr = calculate_atr(df)
    df["ATR"] = atr
    df["ATR_Ratio"] = atr / close.replace(0, np.nan)
    df["MACD_Hist_Ratio"] = macd_hist / close.replace(0, np.nan)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)

    df["BB_PctB"] = (close - bb_lower) / bb_width

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where(
        (plus_dm > minus_dm) & (plus_dm > 0), 0.0
    )
    minus_dm = minus_dm.where(
        (minus_dm > plus_dm) & (minus_dm > 0), 0.0
    )

    atr_safe = atr.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    candle_range = (high - low).replace(0, np.nan)
    df["Upper_Wick_Ratio"] = (
        high - pd.concat([open_price, close], axis=1).max(axis=1)
    ) / candle_range

    df["Lower_Wick_Ratio"] = (
        pd.concat([open_price, close], axis=1).min(axis=1) - low
    ) / candle_range

    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    stoch_range = (highest_high - lowest_low).replace(0, np.nan)
    df["Stoch_K"] = 100 * (close - lowest_low) / stoch_range

    typical_price = (high + low + close) / 3
    typical_sma = typical_price.rolling(20).mean()
    mean_deviation = typical_price.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )
    df["CCI"] = (
        (typical_price - typical_sma)
        / (0.015 * mean_deviation.replace(0, np.nan))
    )

    df["ROC"] = close.pct_change(10)

    # inplace=True を避け、pandas の将来互換性に配慮する。
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# =========================================================
# 市場データ取得・ラベル作成
# =========================================================

def flatten_yf_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]
    return df



def bars(ticker, interval="1h", period="60d"):
    """
    最新の確定足と学習用データを分離して返す。
    学習用データは従来どおり末尾8本を除外する。
    最新足情報は DataFrame.attrs に保持する。
    """
    download_interval = "1h" if interval == "4h" else interval

    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=download_interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=20,
        )
    except Exception as exc:
        print(f"[DATA ERROR] {ticker} {interval}: {exc}")
        return None

    if raw is None or raw.empty:
        print(f"[DATA EMPTY] {ticker} {interval}")
        return None

    raw = flatten_yf_columns(raw)
    required = ["Open", "High", "Low", "Close"]

    if any(col not in raw.columns for col in required):
        print(f"[DATA ERROR] 必要な列がありません: {ticker} {interval}")
        return None

    if "Volume" not in raw.columns:
        raw["Volume"] = 0.0

    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=required)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()

    # timezone-aware timestamps are converted to UTC for a consistent internal index.
    # Naive timestamps are not silently assigned a timezone; hourly freshness checks will hold them.
    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert("UTC")
    else:
        print(f"[TIMEZONE DIAGNOSTIC] {ticker} {interval}: index has no timezone; no implicit conversion")

    # 事前チェック（早期打ち切り用）。最終判定は加工後の本数で行う。
    # 1時間足・日足は末尾の未確定足を1本除外するため +1 本。
    # 4時間足は、完成した4時間足が MIN_FEATURE_BARS 本そろうには
    # 最低でもその4倍の1時間足が必要。
    min_raw_bars = 4 * MIN_FEATURE_BARS if interval == "4h" else MIN_FEATURE_BARS + 1
    if len(raw) < min_raw_bars:
        print(
            f"[DATA SHORT] {ticker} {interval}: {len(raw)} raw bars "
            f"(minimum {min_raw_bars})"
        )
        return None

    if interval == "4h":
        # 4時間足は、終了時刻を過ぎた1時間足だけから作成する。
        # Yahoo Financeの時刻をUTCに統一する。
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        else:
            raw.index = raw.index.tz_convert("UTC")

        # 配信遅延を考慮し、現在時刻より5分前を基準にする。
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5)

        # インデックスを足の開始時刻として扱い、終了済みの1時間足だけ残す。
        raw = raw.loc[
            (raw.index + pd.Timedelta(hours=1)) <= cutoff
        ].copy()

        if raw.empty:
            print(
                f"[DATA SHORT] {ticker} {interval}: "
                "no completed hourly bars"
            )
            return None

        # 4時間区間ごとの元データ本数を数える。
        hourly_counts = raw["Close"].resample(
            "4h", origin="start_day"
        ).count()

        # UTCの4時間区間に集約する。
        raw = raw.resample("4h", origin="start_day").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna(subset=required)

        # 終了済みで、かつ元の1時間足が4本そろった区間だけ採用する。
        complete_counts = hourly_counts.reindex(raw.index).fillna(0)
        raw = raw.loc[
            ((raw.index + pd.Timedelta(hours=4)) <= cutoff)
            & (complete_counts == 4)
        ].copy()
    else:
        # 1時間足・日足は従来どおり最後の取得足を除外する。
        raw = raw.iloc[:-1].copy()

    if len(raw) < MIN_FEATURE_BARS:
        print(
            f"[DATA SHORT] {ticker} {interval}: {len(raw)} bars "
            f"after processing (minimum {MIN_FEATURE_BARS})"
        )
        return None

     # ここで作る指標データは、最新の確定足を含む。
    feature_df = add_indicators(raw)

    # FXのVolumeはデータ元によって全ゼロ・定数になることがある。
    # Vol_Ratioは診断用に残すが、定数になりやすいため学習特徴量から除外する。
    if "Vol_Ratio" in feature_df.columns:
        vol_unique = int(feature_df["Vol_Ratio"].nunique(dropna=True))
        vol_std = safe_float(feature_df["Vol_Ratio"].std(), 0.0)
        print(
            f"[FEATURE DIAGNOSTIC] pair={ticker}, interval={interval}, "
            f"feature=Vol_Ratio, unique_values={vol_unique}, "
            f"std={vol_std}, model_input=False, "
            f"note=diagnostic_only_not_used_for_training"
        )

    feature_df = feature_df.dropna(
        subset=FEATURE_COLUMNS + ["ATR", "Close"]
    )

    if feature_df.empty:
        print(f"[FEATURE EMPTY] {ticker} {interval}")
        return None

    # 学習・予測に使う最新の特徴量と時刻を保持する。
    latest_data = feature_df.iloc[[-1]].copy()
    latest_index = latest_data.index[-1]

    print(
        f"[LATEST CONFIRMED BAR] {ticker} {interval}: "
        f"time_source={latest_index.isoformat()}, "
        f"time_jst={timestamp_diagnostic(latest_index)}, "
        f"close={latest_data['Close'].iloc[-1]}"
    )

    close_values = feature_df["Close"].to_numpy(dtype=float)
    open_values = feature_df["Open"].to_numpy(dtype=float)
    high_values = feature_df["High"].to_numpy(dtype=float)
    low_values = feature_df["Low"].to_numpy(dtype=float)
    atr_values = feature_df["ATR"].to_numpy(dtype=float)

    targets = np.full(len(feature_df), np.nan)

    for i in range(len(feature_df) - LOOKAHEAD_BARS):
        # シグナル確定後の次足始値をエントリー基準にする。
        # 仮想損益と学習ラベルのエントリー時点を一致させる。
        if i + 1 >= len(feature_df):
            continue
        price = open_values[i + 1]
        atr_value = atr_values[i]

        if (
            not np.isfinite(price)
            or not np.isfinite(atr_value)
            or price <= 0
            or atr_value <= 0
        ):
            continue

        upper = price + atr_value
        lower = price - atr_value
        label = 0

        for j in range(i + 1, i + LOOKAHEAD_BARS + 1):
            hit_up = high_values[j] >= upper
            hit_down = low_values[j] <= lower

            if hit_up and hit_down:
                # 学習ラベルでは足内の到達順が不明なケースを中立扱い。
                # 仮想損益では予測方向に対して損切り側を優先する。
                label = 0
                break
            if hit_up:
                label = 1
                break
            if hit_down:
                label = -1
                break

        targets[i] = label

    training_df = feature_df.copy()
    training_df["Target"] = targets
    # 検証専用の仮想損益計算用。モデルの特徴量には絶対に含めない。
    # 学習ラベルと同じく、エントリー時ATRを基準に、今後8本の高値・安値で
    # 利確/損切りの先着を判定できるようにする。どちらにも到達しない場合は8本後終値。
    # シグナル足の確定後に取引する想定で、次の足の始値を記録する。
    # この列は仮想損益検証専用で、モデルの特徴量には含めない。
    training_df["Future_Open_1"] = feature_df["Open"].shift(-1)
    training_df["Future_Close"] = feature_df["Close"].shift(-LOOKAHEAD_BARS)
    future_path_columns = []
    for step in range(1, LOOKAHEAD_BARS + 1):
        high_col = f"Future_High_{step}"
        low_col = f"Future_Low_{step}"
        training_df[high_col] = feature_df["High"].shift(-step)
        training_df[low_col] = feature_df["Low"].shift(-step)
        future_path_columns.extend([high_col, low_col])

    training_df = training_df.dropna(
        subset=FEATURE_COLUMNS + [
            "Target", "ATR", "Close", "Future_Open_1", "Future_Close",
            *future_path_columns
        ]
    )

    if len(training_df) < MIN_TRAIN_ROWS:
        print(
            f"[DATA SHORT AFTER CLEAN] {ticker} {interval}: "
            f"{len(training_df)} rows"
        )
        return None


    training_df["Target"] = training_df["Target"].astype(int)

    # 学習用データと最新足の情報を分離して保持する。
    training_df.attrs["latest_data"] = latest_data
    training_df.attrs["latest_index"] = latest_index
    training_df.attrs["latest_history"] = feature_df.tail(3).copy()
    training_df.attrs["latest_features"] = feature_df.copy()


    constant_features = [
        col for col in FEATURE_COLUMNS
        if training_df[col].nunique(dropna=True) <= 1
    ]
    print(
        f"[FEATURE CONSTANT CHECK] pair={ticker}, interval={interval}, "
        f"constant_model_features={constant_features}, "
        f"count={len(constant_features)}"
    )
    print(
        f"[TRAINING DATA] {ticker} {interval}: "
        f"rows={len(training_df)}, "
        f"last_labeled={training_df.index[-1].isoformat()}, "
        f"last_labeled_jst={timestamp_diagnostic(training_df.index[-1])}, "
        f"latest_confirmed={latest_index.isoformat()}, "
        f"latest_confirmed_jst={timestamp_diagnostic(latest_index)}, "
        f"close={latest_data['Close'].iloc[-1]}"
    )
    print(
        f"[TARGET DISTRIBUTION] {ticker} {interval}: "
        f"{training_df['Target'].value_counts().sort_index().to_dict()}"
    )

    return training_df



# =========================================================
# 時系列検証
def walk_forward_validation(df, pair_name="通貨ペア不明"):
    """時系列検証と、各クラスの予測件数を通貨ペア名付きで記録する。"""

    def failure(reason, folds=0, details=None):
        return {
            "passed": False,
            "reason": reason,
            "balanced_accuracy": None,
            "baseline_balanced_accuracy": None,
            "class_recall_pct": {},
            "folds": folds,
            "fold_details": details or [],
            "class_counts": {},
        }

    def class_counts(values):
        counts = {-1: 0, 0: 0, 1: 0}

        for value in values:
            label = int(value)
            if label in counts:
                counts[label] += 1

        return {
            "sell": counts[-1],
            "neutral": counts[0],
            "buy": counts[1],
        }

    def summarize_trade_results(trades):
        """選択済み取引の損益を集計する。取引の選択処理は行わない。"""
        if not trades:
            return {
                "trades": 0,
                "net_pips": 0.0,
                "win_rate_pct": None,
                "avg_pips_per_trade": None,
                "max_drawdown_pips": 0.0,
            }

        ordered = sorted(trades, key=lambda item: item["index"])
        values = np.array([item["net_pips"] for item in ordered], dtype=float)
        equity = np.cumsum(values)
        peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
        drawdowns = peaks - equity
        return {
            "trades": int(len(values)),
            "net_pips": round(float(values.sum()), 2),
            "win_rate_pct": round(float((values > 0).mean() * 100), 1),
            "avg_pips_per_trade": round(float(values.mean()), 2),
            "max_drawdown_pips": round(float(drawdowns.max(initial=0.0)), 2),
        }

    def simulated_pnl(predictions, test_rows, spread_pips, pip_size):
        """次足始値エントリーによる非重複・検証区間内の概算。実約定成績ではない。"""
        rows = test_rows.sort_index()
        pred_by_index = dict(zip(test_rows.index, predictions))
        trades = []
        if pip_size <= 0 or len(rows) <= LOOKAHEAD_BARS:
            return summarize_trade_results(trades), trades

        # 取引の決済がテスト区間外へはみ出す最後のLOOKAHEAD_BARS本は除外。
        usable_rows = rows.iloc[:-LOOKAHEAD_BARS]
        next_allowed_position = 0

        for position, idx in enumerate(usable_rows.index):
            if position < next_allowed_position:
                continue

            direction = int(pred_by_index.get(idx, 0))
            if direction not in (-1, 1):
                continue  # 中立予測は取引しない

            # シグナル足の終値では約定できないため、次の足の始値を使う。
            # ATRと予測方向はシグナル足で確定した値を使用する。
            entry = safe_float(rows.at[idx, "Future_Open_1"])
            exit_price = safe_float(rows.at[idx, "Future_Close"])
            entry_atr = safe_float(rows.at[idx, "ATR"])
            if entry is None or exit_price is None or entry_atr is None or entry_atr <= 0:
                continue

            # 損益検証を学習ラベルのバリア方式に合わせる。
            # 買い: +1 ATRが利確、-1 ATRが損切り。売りは逆。
            # 同じ足で上下両方に触れた場合、足内の順序が分からないため損切りを先に扱う。
            upper = entry + entry_atr
            lower = entry - entry_atr
            exit_reason = "time_exit"
            for step in range(1, LOOKAHEAD_BARS + 1):
                future_high = safe_float(rows.at[idx, f"Future_High_{step}"])
                future_low = safe_float(rows.at[idx, f"Future_Low_{step}"])
                if future_high is None or future_low is None:
                    exit_price = None
                    break

                hit_upper = future_high >= upper
                hit_lower = future_low <= lower
                if direction == 1:
                    if hit_lower:
                        exit_price = lower
                        exit_reason = "stop_loss_or_ambiguous_bar"
                        break
                    if hit_upper:
                        exit_price = upper
                        exit_reason = "take_profit"
                        break
                else:
                    if hit_upper:
                        exit_price = upper
                        exit_reason = "stop_loss_or_ambiguous_bar"
                        break
                    if hit_lower:
                        exit_price = lower
                        exit_reason = "take_profit"
                        break

            if exit_price is None:
                continue

            gross_pips = (exit_price - entry) * direction / pip_size
            slippage_cost_pips = float(SIMULATED_ROUND_TRIP_SLIPPAGE_PIPS)
            swap_stress_cost_pips = float(SIMULATED_SWAP_STRESS_COST_PIPS)
            total_cost_pips = float(spread_pips) + slippage_cost_pips + swap_stress_cost_pips
            net_pips = gross_pips - total_cost_pips
            adx_value = safe_float(rows.at[idx, "ADX"]) if "ADX" in rows.columns else None
            atr_ratio_value = safe_float(rows.at[idx, "ATR_Ratio"]) if "ATR_Ratio" in rows.columns else None
            if adx_value is None:
                market_regime = "ADX unavailable"
            elif adx_value >= 25.0:
                market_regime = "stronger trend (ADX>=25)"
            else:
                market_regime = "weaker trend/range (ADX<25)"

            trades.append({
                "index": idx,
                "direction": direction,
                "gross_pips": float(gross_pips),
                "spread_cost_pips": float(spread_pips),
                "slippage_cost_pips": slippage_cost_pips,
                "swap_stress_cost_pips": swap_stress_cost_pips,
                "total_cost_pips": total_cost_pips,
                "net_pips": float(net_pips),
                "exit_reason": exit_reason,
                "adx": adx_value,
                "atr_ratio": atr_ratio_value,
                "market_regime": market_regime,
            })
            # 8本後の終値で決済した後まで、新しい取引を開始しない。
            next_allowed_position = position + LOOKAHEAD_BARS

        return summarize_trade_results(trades), trades

    if df is None or len(df) < MIN_TRAIN_ROWS:
        return failure("学習データ不足")

    X = df[FEATURE_COLUMNS]
    y = df["Target"].astype(int)

    if y.nunique() < 2:
        return failure("目的変数の種類が不足")

    fold_size = len(df) // (N_WALK_FORWARD_FOLDS + 1)

    if fold_size < 30:
        return failure("時系列検証データ不足")

    purge_gap = max(0, int(LOOKAHEAD_BARS))

    all_true = []
    all_pred = []
    all_baseline = []
    all_probability_rows = []
    fold_details = []
    simulated_trade_rows = []
    pair_config = PAIRS.get(pair_name, {})
    spread_pips = safe_float(pair_config.get("spread_pips"), 0.0) or 0.0
    pip_size = safe_float(pair_config.get("pip"), 0.0) or 0.0
    used_folds = 0

    # 特徴量重要度を、検証に成功したfoldごとに記録する。
    feature_importance_history = []

    for fold in range(1, N_WALK_FORWARD_FOLDS + 1):
        train_end = fold_size * fold
        test_end = min(
            fold_size * (fold + 1),
            len(df),
        )

        train_cutoff = max(0, train_end - purge_gap)

        train_X = X.iloc[:train_cutoff]
        train_y = y.iloc[:train_cutoff]

        test_X = X.iloc[train_end:test_end]
        test_y = y.iloc[train_end:test_end]

        if len(train_X) < MIN_TRAIN_ROWS or len(test_X) < 10:
            print(
                f"[VALIDATION FOLD SKIP] pair={pair_name}, fold={fold}, "
                f"train_rows={len(train_X)}, "
                f"test_rows={len(test_X)}"
            )
            continue

        if train_y.nunique() < 2:
            print(
                f"[VALIDATION FOLD SKIP] pair={pair_name}, "
                f"fold={fold}: 学習クラス不足"
            )
            continue

        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )

        weights = compute_sample_weight(
            class_weight="balanced",
            y=train_y,
        )

        try:
            model.fit(
                train_X,
                train_y,
                sample_weight=weights,
            )

            pred = model.predict(test_X)
            # RandomForestのpredict_probaは校正済み確率ではない。
            # OOF予測のBrier scoreを記録し、確率値の信頼性を診断する。
            raw_proba = model.predict_proba(test_X)
            proba_by_class = {int(c): raw_proba[:, j] for j, c in enumerate(model.classes_)}
            aligned_proba = np.column_stack([
                proba_by_class.get(-1, np.zeros(len(test_X))),
                proba_by_class.get(0, np.zeros(len(test_X))),
                proba_by_class.get(1, np.zeros(len(test_X))),
            ])
            row_brier = []
            for row_i, actual in enumerate(test_y.to_numpy(dtype=int)):
                one_hot = np.array([1.0 if actual == cls else 0.0 for cls in (-1, 0, 1)])
                row_brier.append(float(np.mean((aligned_proba[row_i] - one_hot) ** 2)))
            fold_brier = float(np.mean(row_brier)) if row_brier else None
            all_probability_rows.extend([
                {"actual": int(actual), "probabilities": aligned_proba[i].tolist()}
                for i, actual in enumerate(test_y.to_numpy(dtype=int))
            ])

            fold_pnl, fold_trades = simulated_pnl(
                pred,
                df.loc[test_X.index],
                spread_pips,
                pip_size,
            )
            # 各foldで選ばれた非重複取引だけを全体集計に使う。
            simulated_trade_rows.extend(fold_trades)

            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:5]
            top_features = [
                (
                    str(train_X.columns[i]),
                    round(float(importances[i]), 4),
                )
                for i in top_idx
            ]

            print(
                f"[FEATURE IMPORTANCE] pair={pair_name}, "
                f"fold={fold}, top5={top_features}"
            )

            cm = confusion_matrix(
                test_y, pred, labels=[-1, 0, 1]
            ).tolist()

            sell_to_buy_rate = (
                cm[0][2] / sum(cm[0]) * 100
                if sum(cm[0]) > 0 else 0.0
            )
            buy_to_sell_rate = (
                cm[2][0] / sum(cm[2]) * 100
                if sum(cm[2]) > 0 else 0.0
            )

            baseline = DummyClassifier(
                strategy="most_frequent"
            )
            baseline.fit(train_X, train_y)
            baseline_pred = baseline.predict(test_X)

            fold_score = balanced_accuracy_score(
                test_y, pred
            )
            fold_base = balanced_accuracy_score(
                test_y, baseline_pred
            )

            fold_recall = recall_score(
                test_y,
                pred,
                labels=[-1, 0, 1],
                average=None,
                zero_division=0,
            )

            recalls = {
                "sell": round(float(fold_recall[0] * 100), 1),
                "neutral": round(float(fold_recall[1] * 100), 1),
                "buy": round(float(fold_recall[2] * 100), 1),
            }

            train_counts = class_counts(train_y)
            true_counts = class_counts(test_y)
            pred_counts = class_counts(pred)

            fold_start = pd.Timestamp(
                test_X.index[0]
            ).isoformat()
            fold_end_time = pd.Timestamp(
                test_X.index[-1]
            ).isoformat()

            detail = {
                "fold": int(fold),
                "train_rows": int(len(train_X)),
                "test_rows": int(len(test_X)),
                "purge_gap": int(purge_gap),
                "test_start": fold_start,
                "test_end": fold_end_time,
                "balanced_accuracy": round(
                    float(fold_score), 4
                ),
                "baseline_balanced_accuracy": round(
                    float(fold_base), 4
                ),
                "class_recall_pct": recalls,
                "train_class_counts": train_counts,
                "actual_class_counts": true_counts,
                "predicted_class_counts": pred_counts,
                "confusion_matrix": cm,
                "brier_score_raw_probability_diagnostic": round(fold_brier, 5) if fold_brier is not None else None,
                "simulated_pnl": fold_pnl,
            }

            fold_details.append(detail)

            print(
                f"[VALIDATION FOLD] pair={pair_name}, fold={fold}, "
                f"test={fold_start}..{fold_end_time}, "
                f"train_rows={len(train_X)}, "
                f"test_rows={len(test_X)}, "
                f"score={fold_score:.4f}, "
                f"baseline={fold_base:.4f}, "
                f"buy_recall={recalls['buy']:.1f}%, "
                f"sell_recall={recalls['sell']:.1f}%, "
                f"train_counts={train_counts}, "
                f"actual_counts={true_counts}, "
                f"predicted_counts={pred_counts}, "
                f"confusion_matrix={cm}, "
                f"sell_to_buy={sell_to_buy_rate:.1f}%, "
                f"buy_to_sell={buy_to_sell_rate:.1f}%, "
                f"raw_proba_brier={round(fold_brier, 5) if fold_brier is not None else None}, "
                f"sim_trades={fold_pnl['trades']}, "
                f"sim_net_pips={fold_pnl['net_pips']}, "
                f"sim_win_rate={fold_pnl['win_rate_pct']}, "
                f"sim_avg_pips={fold_pnl['avg_pips_per_trade']}, "
                f"sim_max_dd_pips={fold_pnl['max_drawdown_pips']}, "
                f"purge_gap={purge_gap}"
            )

            all_true.extend(test_y.tolist())
            all_pred.extend(pred.tolist())
            all_baseline.extend(baseline_pred.tolist())

            # 検証処理を完了したfoldの重要度だけを記録する。
            feature_importance_history.append({
                str(name): float(value)
                for name, value in zip(
                    train_X.columns,
                    importances,
                )
            })
            used_folds += 1

        except Exception as exc:
            print(
                f"[VALIDATION ERROR] pair={pair_name}, fold={fold}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not all_true or used_folds == 0:
        return failure(
            "時系列検証を完了できませんでした",
            used_folds,
            fold_details,
        )

    # fold間での特徴量重要度の平均・ばらつき・トップ5登場回数を集計。
    if feature_importance_history:
        importance_summary = []

        for name in FEATURE_COLUMNS:
            values = [
                item[name]
                for item in feature_importance_history
                if name in item
            ]

            if not values:
                continue

            top5_count = sum(
                1
                for item in feature_importance_history
                if name in sorted(
                    item,
                    key=item.get,
                    reverse=True,
                )[:5]
            )

            importance_summary.append({
                "feature": name,
                "top5_count": int(top5_count),
                "mean_importance": round(
                    float(np.mean(values)), 4
                ),
                "std_importance": round(
                    float(np.std(values)), 4
                ),
            })

        importance_summary.sort(
            key=lambda item: item["mean_importance"],
            reverse=True,
        )

        print(
            f"[FEATURE IMPORTANCE SUMMARY] pair={pair_name}, "
            f"folds={len(feature_importance_history)}, "
            f"features={importance_summary}"
        )

    score = balanced_accuracy_score(
        all_true,
        all_pred,
    )
    raw_probability_brier = None
    if all_probability_rows:
        brier_values = []
        for item in all_probability_rows:
            actual = item["actual"]
            probs = np.asarray(item["probabilities"], dtype=float)
            one_hot = np.array([1.0 if actual == cls else 0.0 for cls in (-1, 0, 1)])
            brier_values.append(float(np.mean((probs - one_hot) ** 2)))
        raw_probability_brier = round(float(np.mean(brier_values)), 5)
        print(
            f"[PROBABILITY RELIABILITY DIAGNOSTIC] pair={pair_name}, "
            f"raw_oof_brier={raw_probability_brier}, "
            f"note=lower_is_better; raw RandomForest probabilities are NOT calibrated and must not be interpreted as true win probabilities"
        )
    baseline_score = balanced_accuracy_score(
        all_true,
        all_baseline,
    )

    recalls = recall_score(
        all_true,
        all_pred,
        labels=[-1, 0, 1],
        average=None,
        zero_division=0,
    )

    recall_pct = {
        "sell": round(float(recalls[0] * 100), 1),
        "neutral": round(float(recalls[1] * 100), 1),
        "buy": round(float(recalls[2] * 100), 1),
    }

    actual_counts = class_counts(all_true)
    predicted_counts = class_counts(all_pred)

    # 各fold内で非重複かつ決済がfold内に収まる取引だけを合算する。
    total_pnl = summarize_trade_results(simulated_trade_rows)

    # 買い・売りを分けて評価する。片方の成績が悪い場合に、その方向だけ保留できるようにする。
    directional_pnl = {}
    for direction, direction_name in ((1, "buy"), (-1, "sell")):
        direction_trades = [
            trade for trade in simulated_trade_rows
            if trade.get("direction") == direction
        ]
        direction_summary = summarize_trade_results(direction_trades)
        directional_pnl[direction_name] = {
            **direction_summary,
            "passed": bool(
                direction_summary["trades"] >= MIN_DIRECTIONAL_PNL_TRADES
                and direction_summary["net_pips"] > 0
            ),
            "minimum_trades": MIN_DIRECTIONAL_PNL_TRADES,
            "reason": (
                "方向別の仮想損益条件を満たしました"
                if direction_summary["trades"] >= MIN_DIRECTIONAL_PNL_TRADES
                and direction_summary["net_pips"] > 0
                else "方向別の仮想取引件数不足、または損益がプラスでないため保留"
            ),
        }
        print(
            f"[DIRECTIONAL PNL GATE] pair={pair_name}, direction={direction_name}, "
            f"trades={direction_summary['trades']}, net_pips={direction_summary['net_pips']}, "
            f"minimum_trades={MIN_DIRECTIONAL_PNL_TRADES}, "
            f"passed={directional_pnl[direction_name]['passed']}"
        )

    # 追加の安定性ゲート：全体損益が一部の期間だけに依存していないか確認する。
    # 3期間中2期間以上がプラス、全体損益がプラス、かつ最低30件の仮想取引を要求。
    # これは概算バックテスト上の保守的な足切りであり、実取引の利益を保証しない。
    fold_pnl_values = []
    for detail in fold_details:
        fold_pnl = detail.get("simulated_pnl") or {}
        fold_net = safe_float(fold_pnl.get("net_pips"))
        fold_trades = safe_float(fold_pnl.get("trades"))
        if fold_net is not None and fold_trades is not None:
            fold_pnl_values.append({"net_pips": fold_net, "trades": int(fold_trades)})

    positive_pnl_folds = sum(1 for item in fold_pnl_values if item["trades"] > 0 and item["net_pips"] > 0)
    required_positive_folds = 2 if len(fold_pnl_values) >= 3 else len(fold_pnl_values)
    pnl_stability_ok = bool(
        len(fold_pnl_values) >= N_WALK_FORWARD_FOLDS
        and positive_pnl_folds >= required_positive_folds
        and total_pnl["trades"] >= 30
        and total_pnl["net_pips"] > 0
    )

    # 既存の精度・基準モデル・方向別再現率の合格条件は緩めない。
    # 不合格時にどの条件を満たしていないか、ログで具体的に分かるようにする。
    model_gate_reasons = []
    if score < 0.40:
        model_gate_reasons.append(f"balanced_accuracy不足({score:.3f}<0.400)")
    if score < baseline_score:
        model_gate_reasons.append(
            f"基準モデル未満({score:.3f}<{baseline_score:.3f})"
        )
    if recall_pct["buy"] < 20.0:
        model_gate_reasons.append(
            f"買い再現率不足({recall_pct['buy']:.1f}%<20.0%)"
        )
    if recall_pct["sell"] < 20.0:
        model_gate_reasons.append(
            f"売り再現率不足({recall_pct['sell']:.1f}%<20.0%)"
        )
    model_validation_ok = not model_gate_reasons

    pnl_gate_reasons = []
    if len(fold_pnl_values) < N_WALK_FORWARD_FOLDS:
        pnl_gate_reasons.append(
            f"損益を評価できた期間が不足({len(fold_pnl_values)}/{N_WALK_FORWARD_FOLDS})"
        )
    if positive_pnl_folds < required_positive_folds:
        pnl_gate_reasons.append(
            f"プラス期間不足({positive_pnl_folds}/{required_positive_folds})"
        )
    if total_pnl["trades"] < 30:
        pnl_gate_reasons.append(
            f"仮想取引件数不足({total_pnl['trades']}件<30件)"
        )
    if total_pnl["net_pips"] <= 0:
        pnl_gate_reasons.append(
            f"仮想損益がプラスでない({total_pnl['net_pips']:.2f} pips)"
        )
    pnl_stability_ok = not pnl_gate_reasons
    passed = bool(model_validation_ok and pnl_stability_ok)

    if not model_validation_ok and not pnl_stability_ok:
        reason = "モデル検証と仮想損益の安定性条件を満たさない"
    elif not model_validation_ok:
        reason = "モデル検証条件を満たさない"
    elif not pnl_stability_ok:
        reason = "仮想損益の期間別安定性条件を満たさない（利益保証ではありません）"
    else:
        reason = "時系列検証と仮想損益の期間別安定性条件を満たしました（実取引成績ではありません）"

    print(
        f"[VALIDATION GATE DETAILS] pair={pair_name}, "
        f"model_gate={'PASS' if model_validation_ok else 'FAIL'}, "
        f"model_reasons={model_gate_reasons or ['none']}, "
        f"pnl_gate={'PASS' if pnl_stability_ok else 'FAIL'}, "
        f"pnl_reasons={pnl_gate_reasons or ['none']}"
    )

    print(
        f"[PNL STABILITY GATE] pair={pair_name}, "
        f"folds_with_pnl={len(fold_pnl_values)}, "
        f"positive_folds={positive_pnl_folds}, "
        f"required_positive_folds={required_positive_folds}, "
        f"total_trades={total_pnl['trades']}, "
        f"total_net_pips={total_pnl['net_pips']}, "
        f"pnl_stability_ok={pnl_stability_ok}, "
        f"model_validation_ok={model_validation_ok}, "
        f"passed={passed}"
    )

    # 売買方向別の損益を出し、買い・売りのどちらが成績を悪化させているか確認する。
    for direction, direction_name in ((1, "BUY"), (-1, "SELL")):
        direction_trades = [t for t in simulated_trade_rows if t.get("direction") == direction]
        direction_summary = summarize_trade_results(direction_trades)
        gross_total = round(sum(t.get("gross_pips", 0.0) for t in direction_trades), 2)
        spread_total = round(sum(t.get("spread_cost_pips", 0.0) for t in direction_trades), 2)
        print(
            f"[SIMULATED PNL BY DIRECTION] pair={pair_name}, direction={direction_name}, "
            f"trades={direction_summary['trades']}, gross_pips={gross_total}, "
            f"spread_cost_pips={spread_total}, net_pips={direction_summary['net_pips']}, "
            f"win_rate={direction_summary['win_rate_pct']}, "
            f"avg_pips={direction_summary['avg_pips_per_trade']}"
        )

    # ADX 25を目安に、トレンドが比較的強い局面と弱い局面を分けて確認する。
    for regime in ("stronger trend (ADX>=25)", "weaker trend/range (ADX<25)", "ADX unavailable"):
        regime_trades = [t for t in simulated_trade_rows if t.get("market_regime") == regime]
        if not regime_trades:
            continue
        regime_summary = summarize_trade_results(regime_trades)
        gross_total = round(sum(t.get("gross_pips", 0.0) for t in regime_trades), 2)
        spread_total = round(sum(t.get("spread_cost_pips", 0.0) for t in regime_trades), 2)
        print(
            f"[SIMULATED PNL BY MARKET REGIME] pair={pair_name}, regime={regime}, "
            f"trades={regime_summary['trades']}, gross_pips={gross_total}, "
            f"spread_cost_pips={spread_total}, net_pips={regime_summary['net_pips']}, "
            f"win_rate={regime_summary['win_rate_pct']}, "
            f"avg_pips={regime_summary['avg_pips_per_trade']}"
        )

    # スプレッド負担が総利益をどの程度押し下げたかを確認する。
    gross_all = round(sum(t.get("gross_pips", 0.0) for t in simulated_trade_rows), 2)
    spread_all = round(sum(t.get("spread_cost_pips", 0.0) for t in simulated_trade_rows), 2)
    print(
        f"[SIMULATED PNL COST CHECK] pair={pair_name}, gross_pips={gross_all}, "
        f"spread_cost_pips={spread_all}, "
        f"slippage_cost_pips={round(sum(t.get('slippage_cost_pips', 0.0) for t in simulated_trade_rows), 2)}, "
        f"swap_stress_cost_pips={round(sum(t.get('swap_stress_cost_pips', 0.0) for t in simulated_trade_rows), 2)}, "
        f"assumed_slippage_pips_per_trade={SIMULATED_ROUND_TRIP_SLIPPAGE_PIPS}, "
        f"assumed_swap_stress_pips_per_trade={SIMULATED_SWAP_STRESS_COST_PIPS}, net_pips={total_pnl['net_pips']}, "
        f"note=next-bar-open entry; spread plus modeled slippage and swap-stress costs deducted; swap value is only a placeholder, not broker data; excludes real fills"
    )

    print(
        f"[SIMULATED PNL SUMMARY] pair={pair_name}, "
        f"trades={total_pnl['trades']}, "
        f"net_pips={total_pnl['net_pips']}, "
        f"win_rate={total_pnl['win_rate_pct']}, "
        f"avg_pips={total_pnl['avg_pips_per_trade']}, "
        f"max_drawdown_pips={total_pnl['max_drawdown_pips']}, "
        f"spread_pips={spread_pips}, hold_bars={LOOKAHEAD_BARS}, "
        f"note=next-bar-open entry; non-overlapping ATR-barrier estimate; same-bar ambiguity counted adversely; includes assumed costs only, not actual broker fills/swaps"
    )

    print(
        f"[VALIDATION DETAIL] pair={pair_name}, "
        f"score={score:.4f}, "
        f"baseline={baseline_score:.4f}, "
        f"buy_recall={recall_pct['buy']:.1f}%, "
        f"sell_recall={recall_pct['sell']:.1f}%, "
        f"actual_counts={actual_counts}, "
        f"predicted_counts={predicted_counts}, "
        f"folds={used_folds}, "
        f"purge_gap={purge_gap}, "
        f"model_validation_ok={model_validation_ok}, "
        f"pnl_stability_ok={pnl_stability_ok}, "
        f"positive_pnl_folds={positive_pnl_folds}, "
        f"passed={passed}"
    )

    return {
        "passed": bool(passed),
        "reason": reason,
        "model_validation_ok": bool(model_validation_ok),
        "model_gate_reasons": model_gate_reasons,
        "pnl_stability_ok": bool(pnl_stability_ok),
        "pnl_gate_reasons": pnl_gate_reasons,
        "positive_pnl_folds": int(positive_pnl_folds),
        "total_pnl_trades": int(total_pnl["trades"]),
        "balanced_accuracy": round(float(score), 4),
        "raw_oof_brier_score_diagnostic": raw_probability_brier,
        "probability_calibrated": False,
        "baseline_balanced_accuracy": round(
            float(baseline_score), 4
        ),
        "class_recall_pct": recall_pct,
        "folds": used_folds,
        "fold_details": fold_details,
        "simulated_pnl": total_pnl,
        "directional_pnl": directional_pnl,
        "class_counts": {
            "actual": actual_counts,
            "predicted": predicted_counts,
        },
    }
def fast_move_detected(df):
    if df is None or len(df) < 4:
        return False, None

    latest = df.iloc[-1]
    atr = safe_float(latest.get("ATR"))

    if atr is None or atr <= 0:
        return False, None

    recent = df.iloc[-3:]
    move = abs(float(recent["Close"].iloc[-1]) - float(recent["Close"].iloc[0]))
    ratio = move / atr

    return ratio >= FAST_MOVE_ATR_MULTIPLE, round(float(ratio), 2)



def build_feature_snapshot(latest_data):
    """シグナル足（最新の確定足）の特徴量を、記録用に辞書化する。

    フォワード検証の記録へ予測時点の材料を残すためだけのもので、
    売買判定には使わない。値が欠けている特徴量は None にする。
    """
    if latest_data is None or latest_data.empty:
        return None
    try:
        row = latest_data.iloc[-1]
    except Exception:
        return None
    snapshot = {}
    for column in FEATURE_COLUMNS:
        snapshot[column] = safe_float(row.get(column))
    return snapshot


def analyze(df, validation):
    if df is None or len(df) < MIN_TRAIN_ROWS:
        return {
            "direction": "様子見",
            "buy_pct": None,
            "sell_pct": None,
            "neutral_pct": None,
            "reason": "学習データ不足",
            "feature_snapshot": None,
        }

    if not validation.get("passed"):
        return {
            "direction": "様子見",
            "buy_pct": None,
            "sell_pct": None,
            "neutral_pct": None,
            "reason": validation.get("reason", "検証条件未達"),
            "feature_snapshot": None,
        }

    latest_data = df.attrs.get("latest_data")
    if latest_data is None or latest_data.empty:
        return {
            "direction": "様子見",
            "buy_pct": None,
            "sell_pct": None,
            "neutral_pct": None,
            "reason": "最新の確定足を確認できません",
            "feature_snapshot": None,
        }

    X = df[FEATURE_COLUMNS]
    y = df["Target"].astype(int)

    if y.nunique() < 2:
        return {
            "direction": "様子見",
            "buy_pct": None,
            "sell_pct": None,
            "neutral_pct": None,
            "reason": "学習対象の種類が不足",
            "feature_snapshot": None,
        }

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y,
    )

    try:
        model.fit(X, y, sample_weight=weights)

        latest_x = latest_data[FEATURE_COLUMNS]
        probabilities = model.predict_proba(latest_x)[0]
        class_to_probability = dict(zip(model.classes_, probabilities))

        sell_pct = float(class_to_probability.get(-1, 0.0) * 100)
        neutral_pct = float(class_to_probability.get(0, 0.0) * 100)
        buy_pct = float(class_to_probability.get(1, 0.0) * 100)

        if buy_pct >= 55 and buy_pct > sell_pct:
            direction = "買い"
        elif sell_pct >= 55 and sell_pct > buy_pct:
            direction = "売り"
        else:
            direction = "様子見"

        return {
            "direction": direction,
            "buy_pct": round(buy_pct, 1),
            "sell_pct": round(sell_pct, 1),
            "neutral_pct": round(neutral_pct, 1),
            "reason": (
                "表示値は未校正のRandomForestモデルスコアです。実際の勝率・真の確率ではありません。"
                "将来の利益を保証するものではありません。"
            ),
            "feature_snapshot": build_feature_snapshot(latest_data),
        }

    except Exception as exc:
        return {
            "direction": "様子見",
            "buy_pct": None,
            "sell_pct": None,
            "neutral_pct": None,
            "feature_snapshot": None,
            "reason": "モデル分析エラー: " + str(exc)[:200],
        }



# =========================================================
# 期間ごとのトレンド
# =========================================================

def timeframe_trend(df):
    if df is None or df.empty:
        return "データ不足"

    # 最新足を含む指標データを優先する
    latest_features = df.attrs.get("latest_features")
    if latest_features is not None and not latest_features.empty:
        source = latest_features
    else:
        source = df

    close = source["Close"]
    if len(close) < 50:
        return "データ不足"

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    last_close = safe_float(close.iloc[-1])
    last_ema20 = safe_float(ema20.iloc[-1])
    last_ema50 = safe_float(ema50.iloc[-1])

    if None in (last_close, last_ema20, last_ema50):
        return "データ不足"

    if last_close > last_ema20 > last_ema50:
        return "上昇"
    if last_close < last_ema20 < last_ema50:
        return "下降"

    return "中立"




def validate_latest_bar(latest_data):
    """最新確定足のOHLC・価格・時刻を簡易検査。異常時は候補を保留する。"""
    if latest_data is None or latest_data.empty:
        return False, "最新の確定足がありません"
    required = ["Open", "High", "Low", "Close"]
    if any(column not in latest_data.columns for column in required):
        return False, "OHLC列が不足しています"
    try:
        row = latest_data.iloc[-1]
        values = {column: safe_float(row[column]) for column in required}
        if any(value is None or value <= 0 for value in values.values()):
            return False, "OHLCに欠損値または0以下の価格があります"
        if values["High"] < max(values["Open"], values["Close"]):
            return False, "高値が始値または終値を下回っています"
        if values["Low"] > min(values["Open"], values["Close"]):
            return False, "安値が始値または終値を上回っています"
        if values["High"] < values["Low"]:
            return False, "高値と安値の関係が不正です"
        index_value = pd.Timestamp(latest_data.index[-1])
        if index_value.tzinfo is None:
            return False, "最新足のタイムゾーンが不明です"
        index_jst = index_value.tz_convert("Asia/Tokyo")
        now_jst_ts = pd.Timestamp.now(tz="Asia/Tokyo")
        if index_jst > now_jst_ts + pd.Timedelta(minutes=10):
            return False, "最新足の時刻が未来になっています"
        return True, "OHLC・価格・時刻の簡易検査に合格"
    except Exception as exc:
        return False, f"最新足の検査に失敗: {type(exc).__name__}"


def stale_data_status(age_minutes):
    """古いデータの原因を断定せず、週末・休場の可能性と平日異常を区別する。"""
    if age_minutes is None:
        return {
            "status": "timestamp_unknown",
            "label": "データ時刻不明",
            "reason": "最新足の時刻を確認できません。",
        }
    now = now_jst()
    if now.weekday() >= 5:
        return {
            "status": "possible_market_closure",
            "label": "週末・市場休場の可能性",
            "reason": "週末のため価格更新が止まっている可能性があります。市場再開後も更新されない場合は取得元を確認してください。",
        }
    if now.weekday() == 0 and now.hour < 8:
        return {
            "status": "possible_market_closure",
            "label": "週明け・市場再開待ちの可能性",
            "reason": "週明けの市場再開前後である可能性があります。更新が始まらない場合は取得元を確認してください。",
        }
    return {
        "status": "stale_on_open_day",
        "label": "データ更新遅延・要確認",
        "reason": "市場が動いている時間帯でも最新足が古い可能性があります。データ取得状況と実行ログを確認してください。",
    }


def data_age_minutes(df):
    """1時間足の開始時刻ではなく、足の確定時刻を基準にデータ経過時間を返す。"""
    if df is None or df.empty:
        print("[DATA AGE DEBUG] df is empty")
        return None

    try:
        latest_index = df.attrs.get("latest_index")

        # 最新足情報がなければ、古い学習用データで代用しない。
        if latest_index is None:
            print("[DATA AGE ERROR] latest_index is missing")
            return None

        last_time = pd.Timestamp(latest_index)

        if last_time.tzinfo is None:
            print("[DATA AGE ERROR] latest_index has no timezone")
            return None

        # Yahoo Financeの1時間足インデックスは通常、足の開始時刻。
        # data_age_minutes() は1時間足に対してのみ使うため、確定時刻を開始時刻+1時間で計算する。
        last_time = last_time.tz_convert("Asia/Tokyo")
        confirmed_close_time = last_time + pd.Timedelta(hours=1)
        now = pd.Timestamp.now(tz="Asia/Tokyo")
        age = (now - confirmed_close_time).total_seconds() / 60

        print(
            f"[DATA AGE DEBUG] latest_bar_start={last_time.isoformat()}, "
            f"latest_confirmed_close={confirmed_close_time.isoformat()}, "
            f"now={now.isoformat()}, age_minutes={age:.1f}"
        )

        # 未来時刻のデータは異常として扱う。
        if age < -10:
            print("[DATA AGE ERROR] latest bar is unexpectedly in the future")
            return None

        return round(max(0.0, age), 1)

    except Exception as exc:
        print(f"[DATA AGE ERROR] {type(exc).__name__}: {exc}")
        return None




# =========================================================
# 通貨ペア分析
# =========================================================

def analyze_pair(name, config):
    ticker = config["ticker"]
    print(f"[START] {name} / {ticker}")

    # ニュースはモデルの売買方向を決めるためではなく、
    # 新規候補を保留するためのリスク確認に使う。
    news = fetch_news(ticker)

    result = {
        "name": name,
        "ticker": ticker,
        "updated_at_jst": iso_now(),
        "decision": "判定保留",
        "direction": "様子見",
        "reason": "",
        "price": None,
        "spread_pips": config["spread_pips"],
        "timeframe_trends": [],
        "fast_move_detected": False,
        "fast_move_atr_ratio": None,
        "data_age_minutes": None,
        "data_freshness_status": "unknown",
        "data_quality_status": "not_checked",
        "data_quality_reason": "",
        "validation": {},
        "signal_bar_time": None,
        "atr": None,
        # フォワード検証への記録可否（ニュース以外の全ゲート通過時に True）。
        # 売買候補の採用判定とは分離した、記録専用のフラグ。
        "forward_recordable": False,
        "feature_snapshot": None,
        "buy_pct": None,
        "sell_pct": None,
        "neutral_pct": None,
        **news,
    }

    try:
        # メイン分析は1時間足。4時間足・日足で方向の偏りを確認。
        # 検証窓を約2か月から約6か月へ拡大し、各期間の評価を安定させる。
        hourly = bars(ticker, interval="1h", period="180d")
        four_hour = bars(ticker, interval="4h", period="90d")
        daily = bars(ticker, interval="1d", period="2y")

        trends = [
            timeframe_trend(hourly),
            timeframe_trend(four_hour),
            timeframe_trend(daily),
        ]
        result["timeframe_trends"] = trends

        
        if hourly is None:
            result["reason"] = "1時間足データ不足または取得失敗"
            result["decision"] = "データ不足・保留"
            return result

        latest_data = hourly.attrs.get("latest_data")
        if latest_data is None or latest_data.empty:
            result["reason"] = "最新の確定足を確認できません"
            result["decision"] = "データ鮮度不明・保留"
            return result

        quality_ok, quality_reason = validate_latest_bar(latest_data)
        result["data_quality_status"] = "pass" if quality_ok else "fail"
        result["data_quality_reason"] = quality_reason
        print(
            f"[DATA QUALITY GATE] pair={name}, status={'PASS' if quality_ok else 'FAIL'}, "
            f"reason={quality_reason}"
        )
        if not quality_ok:
            result["decision"] = "価格データ品質異常・保留"
            result["reason"] = quality_reason + "。データ品質を確認できるまで分析候補を出しません。"
            return result

        result["price"] = safe_float(latest_data["Close"].iloc[-1])
        # フォワード検証の記録用（売買判定には使わない）。
        # シグナル足＝最新の確定足。その時刻とATRを結果に残す。
        result["signal_bar_time"] = pd.Timestamp(latest_data.index[-1]).isoformat()
        result["atr"] = (
            safe_float(latest_data["ATR"].iloc[-1])
            if "ATR" in latest_data.columns
            else None
        )
        result["data_age_minutes"] = data_age_minutes(hourly)

        # 最優先の安全ゲート：古いデータでは重い検証や売買方向の算出をしない。
        # 週末・休場時に過去データから不要な売買候補を作らないための早期終了。
        age = result["data_age_minutes"]
        if age is None:
            result["data_freshness_status"] = "timestamp_unknown"
            result["decision"] = "データ鮮度不明・保留"
            result["reason"] = (
                "最新の確定足の時刻を確認できないため、検証計算を行わず保留します。"
            )
            print(
                f"[DATA FRESHNESS GATE] pair={name}, status=FAIL, "
                "reason=age_unavailable, validation_skipped=True"
            )
            return result

        if age > MAX_DATA_AGE_MINUTES:
            stale_info = stale_data_status(age)
            result["data_freshness_status"] = stale_info["status"]
            result["decision"] = "データ古い・保留"
            result["reason"] = (
                f"最新の確定足から約{age:.0f}分経過しています。"
                f"{stale_info['label']}。{stale_info['reason']}"
                "鮮度制限は安全のため変更せず、検証計算を省略します。"
            )
            print(
                f"[DATA FRESHNESS GATE] pair={name}, status=FAIL, "
                f"age_minutes={age}, max_age_minutes={MAX_DATA_AGE_MINUTES}, "
                "validation_skipped=True"
            )
            return result

        result["data_freshness_status"] = "fresh"
        print(
            f"[DATA FRESHNESS GATE] pair={name}, status=PASS, "
            f"age_minutes={age}, max_age_minutes={MAX_DATA_AGE_MINUTES}"
        )

        # 急変動判定も最新足を使う。
        # 直近3本の値動きと最新ATRを比較する。
        fast_move, fast_ratio = fast_move_detected(
            hourly.attrs.get("latest_history")
        )
        result["fast_move_detected"] = bool(fast_move)
        result["fast_move_atr_ratio"] = fast_ratio

        validation = walk_forward_validation(hourly, pair_name=name)
        result["validation"] = validation

        model_result = analyze(hourly, validation)
        result.update({
            "direction": model_result["direction"],
            "buy_pct": model_result["buy_pct"],
            "sell_pct": model_result["sell_pct"],
            "neutral_pct": model_result["neutral_pct"],
            "feature_snapshot": model_result.get("feature_snapshot"),
        })

        # 方向別の仮想損益が未検証・不良なら、その方向の候補を出さず様子見にする。
        directional_pnl = validation.get("directional_pnl") or {}
        selected_direction = result.get("direction")
        selected_key = "buy" if selected_direction == "買い" else "sell" if selected_direction == "売り" else None
        if selected_key:
            direction_check = directional_pnl.get(selected_key) or {}
            if not direction_check.get("passed", False):
                result["direction"] = "様子見"
                result["directional_pnl_hold"] = True
                result["reason"] = (
                    f"{selected_direction}方向の仮想損益が採用条件を満たしていません。"
                    "方向別の取引件数不足、または損益がプラスでないため、新規候補を保留します。"
                )
            else:
                result["directional_pnl_hold"] = False

        
        # 診断ログ：判定保留や信頼度がNoneになる原因を確認
        print(
            f"[DIAGNOSTIC] {name}: "
            f"hourly_rows={len(hourly)}, "
            f"data_age_minutes={result.get('data_age_minutes')}, "
            f"fast_move={fast_move}, "
            f"fast_move_atr_ratio={fast_ratio}, "
            f"validation_passed={validation.get('passed')}, "
            f"validation_reason={validation.get('reason')}, "
            f"model_reason={model_result.get('reason')}, "
            f"buy_pct={model_result.get('buy_pct')}, "
            f"sell_pct={model_result.get('sell_pct')}"
        )


        # 判定保留条件は、売買候補の判定より先に確認する。
        if fast_move:
            result["decision"] = "急変動・判定保留"
            result["reason"] = (
                f"直近の値動きがATRの{fast_ratio}倍に達しました。"
                "値動きが落ち着くまで新規候補を保留します。"
            )
            return result

        age = result["data_age_minutes"]
        if age is None:
            result["decision"] = "データ鮮度不明・保留"
            result["reason"] = (
                "最新の確定足の時刻を確認できないため保留します。"
                "時刻が確認できるデータを取得できるまで、新規候補に使いません。"
            )
            return result

        if age > MAX_DATA_AGE_MINUTES:
            result["decision"] = "データ古い・保留"
            result["reason"] = (
                f"最新の確定足から約{age:.0f}分経過しているため保留します。"
                "週末や市場休場時は古いデータになることがあります。"
            )
            return result

        if not validation.get("passed"):
            result["decision"] = "検証条件未達・保留"
            result["reason"] = validation.get(
                "reason", "複数期間の時系列検証条件を満たさない"
            )
            return result

        # スプレッドがATRに対して大きすぎる場合は候補を保留。
        latest_data = hourly.attrs.get("latest_data")
        atr_value = (
            safe_float(latest_data["ATR"].iloc[-1])
            if latest_data is not None and not latest_data.empty
            else None
)
        if atr_value is None or atr_value <= 0:
            result["decision"] = "ATR不明・保留"
            result["reason"] = "変動幅を確認できないため保留します。"
            return result

        spread_price = config["spread_pips"] * config["pip"]
        spread_atr_ratio = spread_price / atr_value
        result["spread_atr_ratio"] = round(float(spread_atr_ratio), 4)

        if spread_atr_ratio > 0.15:
            result["decision"] = "スプレッド負担大・保留"
            result["reason"] = "想定スプレッドがATRに対して大きいため保留します。"
            return result

        direction = model_result["direction"]

        # 上位足の方向との食い違い判定（読み取り計算のみ）。
        # ニュース確認より前に計算し、フォワード検証の記録可否にも使う。
        # 判定の文言と優先順位（ニュース保留が先）は従来どおり変えない。
        trend_conflict_reason = None
        if direction == "買い" and "下降" in trends:
            trend_conflict_reason = (
                "上位足に下降トレンドがあるため、新規買い候補を保留します。"
            )
        elif direction == "売り" and "上昇" in trends:
            trend_conflict_reason = (
                "上位足に上昇トレンドがあるため、新規売り候補を保留します。"
            )

        # フォワード検証への記録可否。ニュース以外の全ゲート（鮮度・検証・
        # スプレッド・上位足・方向別）を通過し、方向が買い/売りの場合に
        # True とする。ニュース保留でも記録対象にするための分離であり、
        # 売買候補としての採用条件は一切変えない。
        result["forward_recordable"] = bool(
            direction in ("買い", "売り")
            and trend_conflict_reason is None
            and not result.get("directional_pnl_hold")
        )

        # ニュースの確認結果を候補判定に反映する。
        news_allowed, news_reason = news_allows_candidate(news)
        if not news_allowed:
            result["decision"] = "ニュース確認・判定保留"
            result["reason"] = news_reason
            return result

        # 上位足の方向と大きく食い違う候補は保留。
        if trend_conflict_reason is not None:
            result["decision"] = "上位足と不一致・保留"
            result["reason"] = trend_conflict_reason
            return result

        if direction == "様子見":
            result["decision"] = "方向別検証未達・保留" if result.get("directional_pnl_hold") else "様子見"
            if not result.get("directional_pnl_hold"):
                result["reason"] = model_result["reason"]
            return result

        result["decision"] = "候補あり（手動確認のみ）"
        result["reason"] = (
            "検証・データ・ニュースの各確認条件を通過しました。"
            "これは売買推奨や利益保証ではありません。"
            "松井証券FXへの発注は行いません。"
        )

        return result

    except Exception as exc:
        result["decision"] = "エラー・保留"
        result["reason"] = "分析中のエラー: " + str(exc)[:300]
        print(f"[PAIR ERROR] {name}: {exc}")
        return result


# =========================================================
# スキャン予測のフォワード検証
# =========================================================
# 過去のスキャンが出した候補（方向あり）を保存し、後の実行で実際の価格と
# 照合する「測る仕組み」。売買判定・検証ゲート・鮮度ゲートには一切使わない。


def forward_prediction_id(ticker, signal_bar_time):
    """予測の一意ID。同じシグナル足の候補を二重に記録しないために使う。"""
    return f"{ticker}|{signal_bar_time}"


def _empty_forward_news_buckets():
    """ニュース確認状態別の累積カウンタ（3区分の初期値）。

    区分は verified（記録時にニュース確認を通過）／unverified（ニュース
    未確認のまま記録）／unknown（news_verified フィールドが無い旧形式の
    記録。未確認と断定せず「不明」として分ける）。
    """
    sub = {
        "evaluated_total": 0,
        "hit": 0,
        "miss": 0,
        "not_reached": 0,
        "net_pips_total": 0.0,
        "mae_pips_total": 0.0,
    }
    return {
        "verified": dict(sub),
        "unverified": dict(sub),
        "unknown": dict(sub),
    }


def forward_news_bucket_key(prediction):
    """予測レコードのニュース確認状態から集計区分キーを返す。

    news_verified が True/False の場合だけ確認済み/未確認とし、
    フィールド自体が無い旧形式の記録は unknown（不明）に分ける。
    """
    verified = prediction.get("news_verified")
    if verified is True:
        return "verified"
    if verified is False:
        return "unverified"
    return "unknown"


def compute_forward_stats_by_news_verified(predictions):
    """保存済みの評価済みレコードから、ニュース確認状態別の累積を再計算する。

    区分別カウンタが状態ファイルにまだ無い初回だけ使う。評価済みでも
    保持上限（FORWARD_PREDICTIONS_MAX）でリストから切り詰められた記録は
    再計算に含められない（全体の累積カウンタ側には残っている）。その
    場合、区分別の合計は全体の累積と一致しないことがある。
    """
    buckets = _empty_forward_news_buckets()
    for item in predictions:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if status not in ("HIT", "MISS", "NOT_REACHED"):
            continue
        bucket = buckets[forward_news_bucket_key(item)]
        bucket["evaluated_total"] += 1
        if status == "HIT":
            bucket["hit"] += 1
        elif status == "MISS":
            bucket["miss"] += 1
        else:
            bucket["not_reached"] += 1
        bucket["net_pips_total"] = round(
            float(bucket["net_pips_total"])
            + (safe_float(item.get("net_pips"), 0.0) or 0.0),
            2,
        )
        bucket["mae_pips_total"] = round(
            float(bucket["mae_pips_total"])
            + (safe_float(item.get("mae_pips"), 0.0) or 0.0),
            2,
        )
    return buckets


def _normalize_forward_news_buckets(raw):
    """保存済みの区分別カウンタを防御的に読み直す。"""
    buckets = _empty_forward_news_buckets()
    if not isinstance(raw, dict):
        return buckets
    for name, bucket in buckets.items():
        saved = raw.get(name)
        if not isinstance(saved, dict):
            continue
        for key, default in bucket.items():
            value = saved.get(key, default)
            if isinstance(default, float):
                coerced = safe_float(value)
                bucket[key] = (
                    float(coerced) if coerced is not None else default
                )
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                bucket[key] = int(value)
    return buckets


def new_forward_stats():
    """フォワード検証の累積カウンタ（初期値）。"""
    return {
        "evaluated_total": 0,
        "hit": 0,
        "miss": 0,
        "not_reached": 0,
        "buy_evaluated": 0,
        "buy_hit": 0,
        "buy_miss": 0,
        "sell_evaluated": 0,
        "sell_hit": 0,
        "sell_miss": 0,
        "net_pips_total": 0.0,
        "mae_pips_total": 0.0,
        # ニュース確認状態別の内訳。全体カウンタ（上記）は従来どおり
        # 全区分の合計であり、形式は壊さない。
        "by_news_verified": _empty_forward_news_buckets(),
    }


def normalize_forward_stats(stats):
    """保存済みの累積カウンタを防御的に読み直す。欠けたキーは0で補う。"""
    base = new_forward_stats()
    if not isinstance(stats, dict):
        return base
    for key, default in base.items():
        if key == "by_news_verified":
            base[key] = _normalize_forward_news_buckets(stats.get(key))
            continue
        value = stats.get(key, default)
        if isinstance(default, float):
            coerced = safe_float(value)
            base[key] = float(coerced) if coerced is not None else default
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            base[key] = int(value)
    return base


def _forward_bar_gap_allowed(prev_ts, next_ts):
    """評価ウィンドウ内の隣り合う足の時刻差が許容範囲かを返す。

    許容するのは次の2つだけ。
    - ちょうど1時間（通常の連続した1時間足）
    - 週末の市場休場をまたぐジャンプ: 前の足が金曜（または土曜）で
      次の足が月曜、かつ差が80時間以内
      （実データでは金曜21時→月曜0時の約51〜52時間が通常の週末休場）
    平日の2時間以上の欠けなど、それ以外の差は欠損として扱う。
    時刻の曜日は各タイムスタンプ自身のタイムゾーンで判定する。
    """
    diff = next_ts - prev_ts
    if diff == pd.Timedelta(hours=1):
        return True
    if prev_ts.weekday() in (4, 5) and next_ts.weekday() == 0:
        return diff <= pd.Timedelta(hours=80)
    return False


def evaluate_forward_prediction(prediction, ohlc_df):
    """PENDING予測1件を、シグナル足の直後8本の実際のOHLCで判定する。

    判定の物差しは学習ラベル・仮想損益と同じトリプルバリア（基準価格±1ATR、8本）。
    基準価格は、学習ラベル・仮想損益のエントリー基準に合わせ、シグナル足の
    直後の足（評価ウィンドウの1本目）の始値とする。バリア判定の対象は
    次足自身を含む8本で、学習ラベルの範囲（シグナル後の1〜8本目）と一致する。
    記録に保存された reference_price（シグナル足の終値）は経緯として保持し、
    評価の基準には使わない。評価に使った基準価格は、結果の
    evaluation_reference_price に別途記録する。
    - 方向に有利なバリアに先に到達: HIT（利確価格で決済）
    - 不利なバリアに先に到達: MISS（損切り価格で決済）
    - 同じ足で両方に触れた場合は足内の順序が不明なため、不利側を先として扱う
      （学習ラベルの中立扱いではなく、仮想損益と同じ保守的な扱い）
    - どちらも未到達: NOT_REACHED（8本目の終値で決済）
    損益は既存の仮想損益と同じコスト前提（スプレッド＋スリッページ＋スワップ
    ストレス）を差し引いた純損益pipsも併記する。実約定成績ではない。
    MAEは決済までの足（決済足を含む）の最大不利変動。

    まだ8本そろっていない、途中の足が欠けている（週末の市場休場を除く）、
    または値が不正な場合は None を返し、判定を次回に回す。
    """
    direction = prediction.get("direction")
    sign = 1 if direction == "買い" else -1 if direction == "売り" else 0
    reference = safe_float(prediction.get("reference_price"))
    atr_value = safe_float(prediction.get("atr"))
    pip_size = safe_float(prediction.get("pip_size"))
    spread_pips = safe_float(prediction.get("spread_pips"), 0.0) or 0.0

    if sign == 0 or reference is None or reference <= 0:
        return None
    if atr_value is None or atr_value <= 0 or pip_size is None or pip_size <= 0:
        return None

    try:
        signal_time = pd.Timestamp(prediction.get("signal_bar_time"))
    except Exception:
        return None
    if signal_time is None or signal_time.tzinfo is None:
        return None

    if ohlc_df is None or ohlc_df.empty:
        return None

    frame = ohlc_df.sort_index()
    window = frame[frame.index > signal_time].head(LOOKAHEAD_BARS)
    if len(window) < LOOKAHEAD_BARS:
        return None

    # 足の連続性チェック。bars() 側で欠損行が除かれていると、途中の足が
    # 欠けたまま残りの行で8本そろってしまうことがある。欠けた足を詰めて
    # 評価しないよう、シグナル足→最初の評価足と、ウィンドウ内の連続する
    # 足同士の時刻差を検証し、許容できない欠けがあれば評価せず
    # PENDING のまま残す（週末の通常の市場休場は欠損とみなさない）。
    chain = [signal_time] + list(window.index)
    for prev_ts, next_ts in zip(chain[:-1], chain[1:]):
        if not _forward_bar_gap_allowed(prev_ts, next_ts):
            return None

    # 評価の基準価格は、学習ラベル・仮想損益に合わせて「シグナル足の直後の
    # 足（評価ウィンドウの1本目）の始値」とする。上の連続性チェックにより、
    # ウィンドウの1本目が直後の足であることは保証される（週末をまたぐ場合は
    # 再開後の最初の足）。記録に保存された reference_price（シグナル足の終値）
    # は経緯として保持し、ここでは使わない。
    entry_reference = safe_float(window["Open"].iloc[0])
    if entry_reference is None or entry_reference <= 0:
        return None

    upper = entry_reference + atr_value
    lower = entry_reference - atr_value

    status = "NOT_REACHED"
    exit_price = safe_float(window["Close"].iloc[-1])
    exit_bar_offset = LOOKAHEAD_BARS
    mae_price = 0.0

    for offset, (_, bar) in enumerate(window.iterrows(), start=1):
        high = safe_float(bar["High"])
        low = safe_float(bar["Low"])
        if high is None or low is None:
            return None

        # 決済までの最大不利変動（決済足を含む）。基準は次足始値。
        if sign == 1:
            mae_price = max(mae_price, entry_reference - low)
        else:
            mae_price = max(mae_price, high - entry_reference)

        hit_upper = high >= upper
        hit_lower = low <= lower
        favorable_first = hit_upper if sign == 1 else hit_lower
        adverse_first = hit_lower if sign == 1 else hit_upper

        # 不利側を先に判定する（同一足の両到達はMISS扱い）。
        if adverse_first:
            status = "MISS"
            exit_price = lower if sign == 1 else upper
            exit_bar_offset = offset
            break
        if favorable_first:
            status = "HIT"
            exit_price = upper if sign == 1 else lower
            exit_bar_offset = offset
            break

    if exit_price is None:
        return None

    gross_pips = (exit_price - entry_reference) * sign / pip_size
    cost_pips = (
        float(spread_pips)
        + float(SIMULATED_ROUND_TRIP_SLIPPAGE_PIPS)
        + float(SIMULATED_SWAP_STRESS_COST_PIPS)
    )

    return {
        "status": status,
        # 評価に使った基準価格（次足始値）と、その足の時刻。記録側の
        # reference_price（シグナル足の終値）は上書きしない。
        "evaluation_reference_price": float(entry_reference),
        "entry_bar_time": window.index[0].isoformat(),
        "exit_price": float(exit_price),
        "exit_bar_offset": int(exit_bar_offset),
        "gross_pips": round(float(gross_pips), 2),
        "cost_pips": round(float(cost_pips), 2),
        "net_pips": round(float(gross_pips) - float(cost_pips), 2),
        "mae_pips": round(float(mae_price) / pip_size, 2),
        "evaluated_at_jst": iso_now(),
    }


def update_forward_stats(stats, prediction, outcome):
    """判定済み予測1件を累積カウンタへ反映する。"""
    stats["evaluated_total"] += 1
    status = outcome.get("status")
    if status == "HIT":
        stats["hit"] += 1
    elif status == "MISS":
        stats["miss"] += 1
    elif status == "NOT_REACHED":
        stats["not_reached"] += 1

    if prediction.get("direction") == "買い":
        stats["buy_evaluated"] += 1
        if status == "HIT":
            stats["buy_hit"] += 1
        elif status == "MISS":
            stats["buy_miss"] += 1
    elif prediction.get("direction") == "売り":
        stats["sell_evaluated"] += 1
        if status == "HIT":
            stats["sell_hit"] += 1
        elif status == "MISS":
            stats["sell_miss"] += 1

    net_pips = safe_float(outcome.get("net_pips"), 0.0) or 0.0
    mae_pips = safe_float(outcome.get("mae_pips"), 0.0) or 0.0
    stats["net_pips_total"] = round(float(stats["net_pips_total"]) + float(net_pips), 2)
    stats["mae_pips_total"] = round(float(stats["mae_pips_total"]) + float(mae_pips), 2)

    # ニュース確認状態別の内訳にも同じ1件を反映する（全体と混同しない）。
    buckets = stats.get("by_news_verified")
    if not isinstance(buckets, dict):
        buckets = _empty_forward_news_buckets()
        stats["by_news_verified"] = buckets
    bucket = buckets.setdefault(
        forward_news_bucket_key(prediction),
        _empty_forward_news_buckets()["unknown"],
    )
    bucket["evaluated_total"] = int(bucket.get("evaluated_total", 0)) + 1
    if status == "HIT":
        bucket["hit"] = int(bucket.get("hit", 0)) + 1
    elif status == "MISS":
        bucket["miss"] = int(bucket.get("miss", 0)) + 1
    elif status == "NOT_REACHED":
        bucket["not_reached"] = int(bucket.get("not_reached", 0)) + 1
    bucket["net_pips_total"] = round(
        float(bucket.get("net_pips_total", 0.0)) + float(net_pips), 2
    )
    bucket["mae_pips_total"] = round(
        float(bucket.get("mae_pips_total", 0.0)) + float(mae_pips), 2
    )
    return stats


def build_forward_summary(stats, predictions, evaluated_now, recorded_now,
                          tracking_since=None, recorded_verified_now=0,
                          recorded_unverified_now=0):
    """状態ファイルとサマリー表示用の集計ビューを作る。

    的中率は HIT / (HIT + MISS)。NOT_REACHED（どちらにも届かなかった予測）は
    勝敗が付いていないため分母に入れない。方向別も同じ定義。
    平均MAEは評価済み全件（NOT_REACHED含む）の平均。
    ニュース確認状態別の内訳（news_verified_breakdown）も同じ定義で
    区分ごとに算出し、全体の数値と混同しないように分けて持つ。
    """
    stats = normalize_forward_stats(stats)
    decided = stats["hit"] + stats["miss"]
    buy_decided = stats["buy_hit"] + stats["buy_miss"]
    sell_decided = stats["sell_hit"] + stats["sell_miss"]
    evaluated = stats["evaluated_total"]
    pending = sum(
        1 for item in predictions
        if isinstance(item, dict) and item.get("status") == "PENDING"
    )

    breakdown = {}
    for name, bucket in stats["by_news_verified"].items():
        bucket_decided = bucket["hit"] + bucket["miss"]
        bucket_evaluated = bucket["evaluated_total"]
        breakdown[name] = {
            **bucket,
            "hit_rate_pct": (
                round(bucket["hit"] / bucket_decided * 100, 1)
                if bucket_decided else None
            ),
            "avg_mae_pips": (
                round(float(bucket["mae_pips_total"]) / bucket_evaluated, 2)
                if bucket_evaluated else None
            ),
        }

    return {
        **stats,
        "tracking_since": tracking_since,
        "hit_rate_pct": round(stats["hit"] / decided * 100, 1) if decided else None,
        "buy_hit_rate_pct": (
            round(stats["buy_hit"] / buy_decided * 100, 1) if buy_decided else None
        ),
        "sell_hit_rate_pct": (
            round(stats["sell_hit"] / sell_decided * 100, 1) if sell_decided else None
        ),
        "avg_mae_pips": (
            round(float(stats["mae_pips_total"]) / evaluated, 2) if evaluated else None
        ),
        "news_verified_breakdown": breakdown,
        "pending_count": pending,
        "stored_predictions": len(predictions),
        "evaluated_this_run": evaluated_now,
        "recorded_this_run": recorded_now,
        "recorded_verified_this_run": recorded_verified_now,
        "recorded_unverified_this_run": recorded_unverified_now,
        "definitions": (
            "的中率=HIT/(HIT+MISS)。NOT_REACHEDは勝敗なしとして分母に含めない。"
            "損益は基準価格±1ATRのバリア方式で、既存の仮想損益と同じコスト前提を"
            "差し引いた概算。実約定成績ではない。"
        ),
    }


def process_forward_verification(previous_state):
    """保存済み予測のうち評価期間が終わったものを、実際の価格で判定する。

    ペアごとに1時間足を取り直して判定する。失敗しても例外を外に出さず、
    判定できなかった予測はPENDINGのまま次回に回す。
    戻り値: (predictions, stats, evaluated_now)
    """
    raw_predictions = previous_state.get("predictions", [])
    if not isinstance(raw_predictions, list):
        raw_predictions = []
    predictions = [item for item in raw_predictions if isinstance(item, dict)]
    raw_stats = previous_state.get("forward_stats")
    stats = normalize_forward_stats(raw_stats)

    # ニュース確認状態別の累積がまだ無い状態（この変更より前の形式）では、
    # 初回だけ保存済みレコードから再計算して初期化する。旧形式の記録は
    # news_verified フィールドが無いため unknown（不明）区分に入り、
    # 未確認と断定されることはない。再計算の対象はリスト内に残っている
    # 評価済み記録のみで、切り詰め済みの分は区分別には含められない
    # （全体の累積カウンタには含まれている）。
    if not (isinstance(raw_stats, dict) and "by_news_verified" in raw_stats):
        stats["by_news_verified"] = (
            compute_forward_stats_by_news_verified(predictions)
        )

    pending_by_ticker = {}
    for item in predictions:
        if item.get("status") == "PENDING" and item.get("ticker"):
            pending_by_ticker.setdefault(item["ticker"], []).append(item)

    evaluated_now = 0
    for ticker, pending_items in pending_by_ticker.items():
        try:
            frame = bars(ticker, interval="1h", period="180d")
            if frame is None:
                continue
            ohlc = frame.attrs.get("latest_features")
            if ohlc is None or ohlc.empty:
                continue
            for item in pending_items:
                try:
                    outcome = evaluate_forward_prediction(item, ohlc)
                except Exception as exc:
                    print(
                        f"[FORWARD VERIFY ERROR] id={item.get('id')}: "
                        f"{type(exc).__name__}: {str(exc)[:180]}"
                    )
                    continue
                if outcome is None:
                    continue
                item.update(outcome)
                update_forward_stats(stats, item, outcome)
                evaluated_now += 1
                print(
                    f"[FORWARD VERIFY] id={item.get('id')}, "
                    f"status={outcome['status']}, net_pips={outcome['net_pips']}, "
                    f"mae_pips={outcome['mae_pips']}"
                )
        except Exception as exc:
            print(
                f"[FORWARD VERIFY ERROR] ticker={ticker}: "
                f"{type(exc).__name__}: {str(exc)[:180]}"
            )

    return predictions, stats, evaluated_now


def trim_forward_predictions(predictions, max_items=FORWARD_PREDICTIONS_MAX):
    """予測リストを上限件数に切り詰める。

    未評価（PENDING）の予測は削除しない。超過分は評価済みのものから
    古い順に削除する。PENDING だけで上限を超える場合は全て保持する
    （未評価データを失わない方針。その場合リストは上限より長くなる）。
    リストの時系列順序は維持する。
    """
    if len(predictions) <= max_items:
        return list(predictions)
    excess = len(predictions) - max_items
    kept = []
    for item in predictions:
        is_pending = isinstance(item, dict) and item.get("status") == "PENDING"
        if excess > 0 and not is_pending:
            excess -= 1
            continue
        kept.append(item)
    return kept


def record_forward_predictions(predictions, pair_results, details=None):
    """今回のスキャンで候補（方向あり）になった予測を記録する。

    記録対象は次のいずれか。
    - 全ゲートを通過して「候補あり（手動確認のみ）」になったもの
      （記録時点でニュース確認を通過しているため news_verified=True）
    - ニュース確認だけを理由に保留されたが、ニュース以外の全ゲートを
      通過したもの（forward_recordable=True。news_verified=False）
    ニュース以外のゲート（鮮度・検証等）で保留になった方向は記録しない。
    記録と売買候補の採用は分離しており、採用条件は従来どおり。
    IDは ticker + シグナル足時刻で、記録対象が広がっても同じ候補の
    二重記録を防ぐ。
    details が渡された場合、内訳（recorded_verified /
    recorded_unverified）を書き込む。
    戻り値: 新規記録件数
    """
    existing_ids = {
        item.get("id") for item in predictions if isinstance(item, dict)
    }
    recorded = 0
    recorded_verified = 0
    recorded_unverified = 0
    for item in pair_results:
        try:
            is_candidate = (
                item.get("decision") == "候補あり（手動確認のみ）"
            )
            if not is_candidate and not item.get("forward_recordable"):
                continue
            direction = item.get("direction")
            if direction not in ("買い", "売り"):
                continue
            ticker = item.get("ticker")
            signal_time = item.get("signal_bar_time")
            reference_price = safe_float(item.get("price"))
            atr_value = safe_float(item.get("atr"))
            if not ticker or not signal_time:
                continue
            if reference_price is None or reference_price <= 0:
                continue
            if atr_value is None or atr_value <= 0:
                continue

            pred_id = forward_prediction_id(ticker, signal_time)
            if pred_id in existing_ids:
                continue

            config = PAIRS.get(item.get("name"), {})
            pip_size = safe_float(config.get("pip"), 0.0) or 0.0
            if pip_size <= 0:
                continue

            # 記録時点のニュース確認状態。候補あり＝確認通過（True）。
            # forward_recordable だけで記録される場合は未確認（False）。
            news_verified = bool(is_candidate)

            predictions.append({
                "id": pred_id,
                "recorded_at_jst": iso_now(),
                "name": item.get("name"),
                "ticker": ticker,
                "direction": direction,
                "direction_sign": 1 if direction == "買い" else -1,
                "reference_price": float(reference_price),
                "atr": float(atr_value),
                "signal_bar_time": signal_time,
                "pip_size": float(pip_size),
                "spread_pips": safe_float(config.get("spread_pips"), 0.0) or 0.0,
                "status": "PENDING",
                "news_status_at_record": item.get("news_status"),
                "news_verified": news_verified,
                # 予測時点の材料（後日の要因分析用）。記録専用で、
                # 売買判定・検証ゲート・フォワード評価には使わない。
                "model_version": MODEL_VERSION,
                "feature_snapshot": item.get("feature_snapshot"),
            })
            existing_ids.add(pred_id)
            recorded += 1
            if news_verified:
                recorded_verified += 1
            else:
                recorded_unverified += 1
            print(
                f"[FORWARD RECORD] id={pred_id}, direction={direction}, "
                f"reference={reference_price}, atr={atr_value}, "
                f"news_verified={news_verified}"
            )
        except Exception as exc:
            print(
                f"[FORWARD RECORD ERROR] pair={item.get('name')}: "
                f"{type(exc).__name__}: {str(exc)[:180]}"
            )
    if details is not None:
        details["recorded_verified"] = recorded_verified
        details["recorded_unverified"] = recorded_unverified
    return recorded


# =========================================================
# 全体実行・結果保存
# =========================================================

def write_step_summary(started_at, pair_results, summary_lines,
                       news_status_counts, run_health, forward_summary):
    """GitHub Actions のジョブサマリー（GITHUB_STEP_SUMMARY）へ Markdown を追記する。

    Actions 実行時のみ（環境変数 GITHUB_STEP_SUMMARY がある場合のみ）動作し、
    ローカル実行時は何もしない。表示だけの追加であり、売買判定・検証ゲート・
    鮮度ゲート・ニュース判定には一切使わない。書き出しに失敗しても例外を
    外に出さず、スキャン本体に影響しない。
    """
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    def cell(value):
        # Markdown の表が崩れないよう、パイプと改行を取り除いて1セルに収める。
        text = "-" if value is None else str(value)
        return text.replace("|", "｜").replace("\n", " ").replace("\r", " ")

    try:
        lines = []
        lines.append("## FX Cloud Scan 実行サマリー")
        lines.append("")
        lines.append(f"- 実行日時（JST）: {cell(started_at)}")
        lines.append("")
        lines.append("### ペア別結果")
        lines.append("")
        lines.append("| ペア | 判定 | 方向 | 買い% / 売り% | データ鮮度 | ニュース |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for item in pair_results:
            lines.append(
                "| {name} | {decision} | {direction} | {buy} / {sell} | {fresh} | {news} |".format(
                    name=cell(item.get("name")),
                    decision=cell(item.get("decision")),
                    direction=cell(item.get("direction")),
                    buy=cell(item.get("buy_pct")),
                    sell=cell(item.get("sell_pct")),
                    fresh=cell(item.get("data_freshness_status")),
                    news=cell(item.get("news_status")),
                )
            )
        lines.append("")
        lines.append("### ニュース取得状況")
        lines.append("")
        if news_status_counts:
            for status, count in sorted(news_status_counts.items()):
                lines.append(f"- {cell(status)}: {count}件")
        else:
            lines.append("- 記録なし")
        # 系統別の成功・失敗（ペア単位で集計）。系統ごとに失敗を隔離して
        # いるため、1系統の障害が全体の失敗に見えないように分けて出す。
        # 中央銀行フィードの一部成功は「部分成功」として区別し、系統全体を
        # 単に失敗とは数えない。status を持たない旧形式の記録は error の
        # 有無から従来どおり判定する。
        source_tally = {}
        for item in pair_results:
            for entry in item.get("news_sources") or []:
                if not isinstance(entry, dict):
                    continue
                tally = source_tally.setdefault(
                    entry.get("source") or "不明",
                    {"ok": 0, "partial": 0, "ng": 0},
                )
                status = entry.get("status") or (
                    "error" if entry.get("error") else "ok"
                )
                if status == "partial":
                    tally["partial"] += 1
                elif status == "ok":
                    tally["ok"] += 1
                else:
                    tally["ng"] += 1
        if source_tally:
            parts = []
            for name, tally in sorted(source_tally.items()):
                part = f"{name} 成功{tally['ok']}"
                if tally["partial"]:
                    part += f"・部分成功{tally['partial']}"
                part += f"・失敗{tally['ng']}"
                parts.append(part)
            lines.append(f"- 取得系統別（ペア数）: {cell('、'.join(parts))}")
        lines.append("")
        lines.append("### フォワード検証（予測の事後測定。売買判定には使用しない）")
        lines.append("")
        lines.append(
            "- 今回の新規記録: {total}件"
            "（ニュース確認済み {verified}件・未確認 {unverified}件）".format(
                total=forward_summary.get("recorded_this_run"),
                verified=forward_summary.get("recorded_verified_this_run"),
                unverified=forward_summary.get("recorded_unverified_this_run"),
            )
        )
        lines.append(f"- 今回の評価: {forward_summary.get('evaluated_this_run')}件")
        lines.append(f"- 未評価（PENDING）: {forward_summary.get('pending_count')}件")
        lines.append(
            "- 評価済み累計（全体）: {total}件"
            "（的中 {hit}・不的中 {miss}・未到達 {not_reached}）".format(
                total=forward_summary.get("evaluated_total"),
                hit=forward_summary.get("hit"),
                miss=forward_summary.get("miss"),
                not_reached=forward_summary.get("not_reached"),
            )
        )
        hit_rate = forward_summary.get("hit_rate_pct")
        lines.append(
            "- 的中率（全体）: {rate}（定義: HIT / (HIT + MISS)。"
            "未到達 NOT_REACHED は勝敗なしとして分母に含めない）".format(
                rate=f"{hit_rate}%" if hit_rate is not None else "-",
            )
        )
        lines.append(
            f"- 純損益累計（全体）: {forward_summary.get('net_pips_total')} pips"
            "（仮想損益と同じコスト前提で差し引いた概算。実約定成績ではない）"
        )
        lines.append(f"- 平均MAE（全体）: {forward_summary.get('avg_mae_pips')} pips")
        # ニュース確認状態別の内訳。確認済み・未確認・旧形式（不明）を
        # 全体の数値と混同しないよう、区分ごとに分けて出す。
        breakdown = forward_summary.get("news_verified_breakdown") or {}
        bucket_labels = (
            ("verified", "ニュース確認済み"),
            ("unverified", "ニュース未確認"),
            ("unknown", "確認状態不明（旧形式）"),
        )
        for key, label in bucket_labels:
            bucket = breakdown.get(key) or {}
            bucket_rate = bucket.get("hit_rate_pct")
            lines.append(
                "- 評価済み累計（{label}）: {total}件"
                "（的中 {hit}・不的中 {miss}・未到達 {not_reached}）"
                "／的中率 {rate}／純損益 {net} pips".format(
                    label=label,
                    total=bucket.get("evaluated_total"),
                    hit=bucket.get("hit"),
                    miss=bucket.get("miss"),
                    not_reached=bucket.get("not_reached"),
                    rate=f"{bucket_rate}%" if bucket_rate is not None else "-",
                    net=bucket.get("net_pips_total"),
                )
            )
        lines.append(f"- 記録開始: {forward_summary.get('tracking_since') or '未記録'}")
        lines.append("")
        lines.append("### 実行健康診断（主要項目）")
        lines.append("")
        lines.append(f"- 対象ペア: {run_health.get('pairs_total')}件")
        lines.append(f"- データ鮮度OK: {run_health.get('fresh_data_pairs')}件 / 鮮度NG・不明: {run_health.get('stale_or_unknown_pairs')}件")
        lines.append(f"- データ品質NG: {run_health.get('data_quality_fail_pairs')}件")
        lines.append(f"- 検証合格: {run_health.get('validation_pass_pairs')}件")
        lines.append(f"- 候補あり: {run_health.get('candidate_pairs')}件")
        lines.append("")
        lines.append("### サマリー詳細")
        lines.append("")
        for line in summary_lines:
            lines.append(f"- {cell(line)}")
        lines.append("")

        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print("ジョブサマリーへ実行サマリーを書き出しました。")
    except Exception as exc:
        print(f"[STEP SUMMARY ERROR] {type(exc).__name__}: {str(exc)[:180]}")


STATUS_FILE = "scan_status.json"
STATUS_SCHEMA_VERSION = 1

# ステータス記録へ転記する validation のキー（既存の集計値のみ）。
_STATUS_VALIDATION_KEYS = (
    "passed",
    "reason",
    "model_validation_ok",
    "model_gate_reasons",
    "pnl_stability_ok",
    "pnl_gate_reasons",
    "positive_pnl_folds",
    "total_pnl_trades",
    "balanced_accuracy",
    "baseline_balanced_accuracy",
    "class_recall_pct",
)

# ステータス記録へ転記するフォワード集計のキー（forward_summary の既存値のみ）。
_STATUS_FORWARD_KEYS = (
    "tracking_since",
    "evaluated_total",
    "hit",
    "miss",
    "not_reached",
    "hit_rate_pct",
    "buy_hit_rate_pct",
    "sell_hit_rate_pct",
    "net_pips_total",
    "avg_mae_pips",
    "pending_count",
    "stored_predictions",
    "evaluated_this_run",
    "recorded_this_run",
    "recorded_verified_this_run",
    "recorded_unverified_this_run",
    "news_verified_breakdown",
    "definitions",
)

# ステータス記録へ転記する run_health のキー（既存の集計値のみ）。
_STATUS_RUN_HEALTH_KEYS = (
    "pairs_total",
    "fresh_data_pairs",
    "stale_or_unknown_pairs",
    "data_quality_fail_pairs",
    "validation_pass_pairs",
    "validation_skipped_or_not_passed_pairs",
    "news_unavailable_pairs",
    "news_retrieval_error_pairs",
    "news_empty_result_pairs",
    "news_no_recent_article_pairs",
    "news_recent_articles_checked_pairs",
    "candidate_pairs",
    "execution_note",
)


def build_status_record(started_at, finished_at, pair_results, forward_summary,
                        run_health):
    """日次レビュー用のステータス記録を、既存の集計値から組み立てる。

    ジョブサマリー（write_step_summary）と同じ材料を使う記録専用の追加で、
    新しい計算や判定は一切しない。転記するキーは上記のホワイトリストに
    限定し、ペア結果や集計に含まれる他の情報（価格の詳細など）は
    記録しない。実行メタ情報は GitHub Actions の環境変数がある場合のみ
    取り込む（ローカル実行時は None）。
    """
    pairs = []
    for item in pair_results or []:
        if not isinstance(item, dict):
            continue
        validation = item.get("validation") or {}
        pairs.append({
            "name": item.get("name"),
            "ticker": item.get("ticker"),
            "decision": item.get("decision"),
            "direction": item.get("direction"),
            "validation": {
                key: validation.get(key) for key in _STATUS_VALIDATION_KEYS
            },
            "data_freshness_status": item.get("data_freshness_status"),
            "data_age_minutes": item.get("data_age_minutes"),
            "data_quality_status": item.get("data_quality_status"),
            "news_status": item.get("news_status"),
        })

    forward = {
        key: (forward_summary or {}).get(key) for key in _STATUS_FORWARD_KEYS
    }

    run_number_raw = os.environ.get("GITHUB_RUN_NUMBER")
    run_number = (
        int(run_number_raw)
        if run_number_raw and run_number_raw.isdigit()
        else None
    )

    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "generated_at_jst": finished_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "run": {
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_number": run_number,
            "commit_sha": os.environ.get("GITHUB_SHA"),
        },
        "pairs": pairs,
        "forward": forward,
        "run_health": {
            key: (run_health or {}).get(key) for key in _STATUS_RUN_HEALTH_KEYS
        },
    }


def write_status_file(record, path=STATUS_FILE):
    """ステータス記録を JSON ファイルへ書き出す。

    状態ファイル（scan_state.json）とは別ファイルであり、状態の保存・
    復元には一切関与しない。書き込みは一時ファイル経由の置き換えで
    行い、途中で失敗しても既存のステータスファイルを壊さない。
    """
    target = os.path.abspath(path)
    tmp_path = target + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=json_default)
        f.write("\n")
    os.replace(tmp_path, target)
    return target


def write_status_safely(started_at, pair_results, forward_summary, run_health):
    """ステータス記録の生成と書き出しを行い、成否を返す。

    記録専用の追加であり、失敗しても例外を外に出さず、スキャン本体・
    状態保存・ジョブサマリーに影響させない。
    """
    try:
        record = build_status_record(
            started_at, iso_now(), pair_results, forward_summary, run_health
        )
        write_status_file(record)
        print(f"[STATUS RECORD] path={STATUS_FILE}")
        return True
    except Exception as exc:
        print(
            f"[STATUS RECORD WARNING] {type(exc).__name__}: {str(exc)[:180]}"
        )
        return False


def main():
    started_at = iso_now()
    previous_state = load_state()

    # フォワード検証（測る仕組みの追加。売買判定には使わない）:
    # 前回までの予測のうち評価期間が終わったものを、実際の価格で判定する。
    # 失敗してもスキャン本体を止めない。状態が壊れていても保存済みの値は
    # 可能な限り保持し、履歴を巻き戻さない。
    forward_evaluated_now = 0
    try:
        predictions, forward_stats, forward_evaluated_now = (
            process_forward_verification(previous_state)
        )
    except Exception as exc:
        print(
            f"[FORWARD VERIFY ERROR] {type(exc).__name__}: {str(exc)[:180]}"
        )
        raw_predictions = previous_state.get("predictions")
        predictions = raw_predictions if isinstance(raw_predictions, list) else []
        forward_stats = normalize_forward_stats(previous_state.get("forward_stats"))

    pair_results = []
    for name, config in PAIRS.items():
        pair_results.append(analyze_pair(name, config))

    # 今回の候補をフォワード検証の対象として記録する（判定には使わない）。
    forward_recorded_now = 0
    forward_record_details = {}
    try:
        forward_recorded_now = record_forward_predictions(
            predictions, pair_results, details=forward_record_details
        )
        # 切り詰めは未評価（PENDING）を落とさない方式（trim_forward_predictions）。
        predictions = trim_forward_predictions(predictions)
    except Exception as exc:
        print(
            f"[FORWARD RECORD ERROR] {type(exc).__name__}: {str(exc)[:180]}"
        )

    # 記録の開始日時。キャッシュ消失で状態が作り直されると、この日時が
    # 新しいものにリセットされるため、「記録が失われた」ことと「成績が悪い」
    # ことを区別できる。保存済みの値があればそれを引き継ぐ。
    forward_tracking_since = previous_state.get("forward_tracking_since")
    if not isinstance(forward_tracking_since, str) or not forward_tracking_since:
        recorded_times = [
            item.get("recorded_at_jst")
            for item in predictions
            if isinstance(item, dict) and item.get("recorded_at_jst")
        ]
        forward_tracking_since = min(recorded_times) if recorded_times else None

    forward_summary = build_forward_summary(
        forward_stats,
        predictions,
        forward_evaluated_now,
        forward_recorded_now,
        tracking_since=forward_tracking_since,
        recorded_verified_now=forward_record_details.get("recorded_verified", 0),
        recorded_unverified_now=forward_record_details.get(
            "recorded_unverified", 0
        ),
    )

    summary = []
    for item in pair_results:
        validation = item.get("validation") or {}
        recall = validation.get("class_recall_pct") or {}

        summary.append(
            f"{item['name']}: {item['decision']} / {item['direction']} / "
            f"買い{item.get('buy_pct')}%・売り{item.get('sell_pct')}% / "
            f"ニュース={item.get('news_status')} / "
            f"重要ニュース={item.get('news_risk_detected')} / "
            f"再現率={recall}"
        )

    summary.append(
        f"フォワード検証（記録開始: {forward_summary['tracking_since'] or '未記録'}）: "
        f"評価済み累計{forward_summary['evaluated_total']}件"
        f"（的中{forward_summary['hit']}・不的中{forward_summary['miss']}・"
        f"未到達{forward_summary['not_reached']}） / "
        f"的中率={forward_summary['hit_rate_pct']}%"
        f"（定義: HIT/(HIT+MISS)、未到達は分母外） / "
        f"純損益累計={forward_summary['net_pips_total']}pips / "
        f"平均MAE={forward_summary['avg_mae_pips']}pips / "
        f"未評価={forward_summary['pending_count']}件 / "
        f"今回判定={forward_summary['evaluated_this_run']}件・"
        f"今回記録={forward_summary['recorded_this_run']}件"
        f"（ニュース確認済み{forward_summary['recorded_verified_this_run']}件・"
        f"未確認{forward_summary['recorded_unverified_this_run']}件）"
    )

    history = previous_state.get("history", [])
    if not isinstance(history, list):
        history = []

    # 履歴は軽量な要約だけ保存し、各回の詳細な検証データでJSONが肥大化するのを防ぐ。
    compact_pairs = []
    for item in pair_results:
        validation = item.get("validation") or {}
        simulated_pnl = validation.get("simulated_pnl") or {}
        compact_pairs.append({
            "name": item.get("name"),
            "ticker": item.get("ticker"),
            "decision": item.get("decision"),
            "direction": item.get("direction"),
            "price": item.get("price"),
            "data_age_minutes": item.get("data_age_minutes"),
            "data_freshness_status": item.get("data_freshness_status"),
            "data_quality_status": item.get("data_quality_status"),
            "reason": item.get("reason"),
            "buy_pct": item.get("buy_pct"),
            "sell_pct": item.get("sell_pct"),
            "news_status": item.get("news_status"),
            "validation_passed": validation.get("passed"),
            "model_validation_ok": validation.get("model_validation_ok"),
            "pnl_stability_ok": validation.get("pnl_stability_ok"),
            "simulated_net_pips": simulated_pnl.get("net_pips"),
            "simulated_trades": simulated_pnl.get("trades"),
            "directional_pnl": validation.get("directional_pnl"),
        })

    run_record = {
        "run_at_jst": started_at,
        "summary": summary,
        "pairs": compact_pairs,
    }
    history.append(run_record)
    history = history[-50:]

    news_status_counts = {}
    for item in pair_results:
        status = item.get("news_status", "unknown")
        news_status_counts[status] = news_status_counts.get(status, 0) + 1

    # 実行単位の健康診断。実行成功と分析可能性を分けて見える化する。
    news_health = news_health_summary(pair_results)
    previous_comparison = compare_previous_run(previous_state, pair_results)
    run_health = {
        "pairs_total": len(pair_results),
        "fresh_data_pairs": sum(1 for item in pair_results if item.get("data_freshness_status") == "fresh"),
        "stale_or_unknown_pairs": sum(1 for item in pair_results if item.get("data_freshness_status") != "fresh"),
        "data_quality_fail_pairs": sum(1 for item in pair_results if item.get("data_quality_status") == "fail"),
        "validation_pass_pairs": sum(1 for item in pair_results if (item.get("validation") or {}).get("passed") is True),
        "validation_skipped_or_not_passed_pairs": sum(1 for item in pair_results if (item.get("validation") or {}).get("passed") is not True),
        # 旧フィールド名は互換性のため残し、意味は「直近記事を確認できなかったペア」。
        "news_unavailable_pairs": sum(1 for item in pair_results if item.get("news_status") not in {
            "recent_articles_no_risk_keywords", "recent_articles_with_risk_keywords"
        }),
        "news_retrieval_error_pairs": news_health["retrieval_error_pairs"],
        "news_empty_result_pairs": news_health["empty_result_pairs"],
        "news_no_recent_article_pairs": news_health["articles_returned_but_none_recent_pairs"],
        "news_recent_articles_checked_pairs": news_health["recent_articles_checked_pairs"],
        "candidate_pairs": sum(1 for item in pair_results if item.get("decision") == "候補あり（手動確認のみ）"),
        "execution_note": "スキャン完了は売買検証成功を意味しません。候補数が0でも安全ゲートによる正常な保留の場合があります。",
    }

    state = {
        "schema_version": MODEL_VERSION,
        "scanner_version": f"v{MODEL_VERSION}",
        "last_run_jst": started_at,
        "mode": "v19_entry_aligned_labels_cost_stress_probability_diagnostics_news_health_timezone_feature_diagnostics_data_quality_freshness_directional_pnl_walk_forward_no_auto_orders",
        "interval": "1h",
        "data_freshness_policy": {
            "max_age_minutes": MAX_DATA_AGE_MINUTES,
            "notice": "古いデータでは時系列検証・売買方向の算出を省略し、判定保留にします。制限時間は緩めません。",
            "stale_reason_classification": "週末・休場の可能性と市場稼働日における更新遅延を区別します。原因を確定するものではありません。",
        },
        "directional_pnl_policy": {
            "minimum_trades_per_direction": MIN_DIRECTIONAL_PNL_TRADES,
            "require_positive_net_pips": True,
            "notice": "買い・売りを別々に評価し、件数不足または仮想損益がプラスでない方向は候補を保留します。仮想検証であり将来利益を保証しません。",
        },
        "news_policy": {
            "source": "Googleニュース / 中央銀行公式フィード / Yahoo Finance",
            "max_age_hours": NEWS_MAX_AGE_HOURS,
            "hold_on_important_news": NEWS_HOLD_ON_IMPORTANT,
            "hold_if_news_unavailable": NEWS_HOLD_IF_UNAVAILABLE,
            "notice": (
                "ニュース見出しを売買方向の判断には使用しない。"
                "重要キーワード検出時やニュース確認不能時は新規候補を保留する。"
                "公式経済指標カレンダーの代替ではない。"
            ),
        },
        "predictions": predictions,
        "forward_stats": forward_stats,
        "forward_tracking_since": forward_tracking_since,
        "forward_verification": forward_summary,
        "forward_verification_policy": {
            "lookahead_bars": LOOKAHEAD_BARS,
            "barrier": "基準価格±1ATR（学習ラベル・仮想損益と同じ物差し）",
            "same_bar_both_hit": "足内の順序が不明なため不利側を先として扱う",
            "hit_rate_definition": "HIT/(HIT+MISS)。NOT_REACHEDは分母に含めない",
            "cost_model": "スプレッド＋スリッページ＋スワップストレス（仮想損益と同じ前提）。実約定成績ではない",
            "max_stored_predictions": FORWARD_PREDICTIONS_MAX,
            "record_scope": (
                "候補あり（ニュース確認済み）に加え、ニュース確認だけを理由に"
                "保留されてもニュース以外の全条件を満たした予測も記録する。"
                "記録には news_verified（確認済み/未確認）を持たせ、累積統計は"
                "全体と確認状態別（by_news_verified）を分けて保持する。"
                "news_verified フィールドが無い旧形式の記録は「不明」区分とし、"
                "未確認とは断定しない。売買候補の採用条件は従来どおり。"
            ),
            "notice": (
                "状態はActionsキャッシュで引き継ぐため、長期間未使用などで"
                "キャッシュが消えると記録はリセットされる。その場合は"
                "forward_tracking_since が新しい日時になる。"
            ),
        },
        "news_status_counts": news_status_counts,
        "news_health": news_health,
        "previous_run_comparison": previous_comparison,
        "run_health": run_health,
        "data_quality_policy": {
            "checks": ["OHLC欠損・非正価格", "高値安値と始値終値の整合性", "タイムゾーン不明", "未来時刻"],
            "notice": "簡易検査であり、価格配信元の正確性や全ての異常を保証するものではありません。",
        },
        "summary": summary,
        "pairs": pair_results,
        "history": history,
        "history_policy": {
            "max_runs": 50,
            "detail": "履歴には軽量な要約のみを保存。詳細な検証値は最新実行のpairsに保存。",
        },
        "safety_notice": (
            "価格分析・注文補助専用です。証券口座接続・自動発注・自動決済は行いません。"
            "分析結果は将来の利益を保証しません。"
        ),
    }

    save_state(state)

    # GitHub Actions のジョブサマリーへ表示用サマリーを書き出す
    # （表示のみの追加。書き出し失敗でもスキャン本体に影響しない）。
    write_step_summary(
        started_at, pair_results, summary,
        news_status_counts, run_health, forward_summary,
    )

    # 日次レビュー用のステータス記録（scan_status.json）を書き出す。
    # 記録専用の追加で、失敗してもスキャン本体に影響しない。
    write_status_safely(started_at, pair_results, forward_summary, run_health)

    print("\n========== FX SCAN SUMMARY ==========")
    print("\n".join(summary))
    print("ニュース取得状況:", news_status_counts)
    print("ニュース診断詳細:", json.dumps(news_health, ensure_ascii=False))
    print("前回実行との比較:", json.dumps(previous_comparison, ensure_ascii=False))
    print("フォワード検証:", json.dumps(forward_summary, ensure_ascii=False))
    print("実行健康診断:", json.dumps(run_health, ensure_ascii=False))
    print("通貨ペア別理由:")
    for item in pair_results:
        print(
            f"  - {item.get('name')}: decision={item.get('decision')}; "
            f"freshness={item.get('data_freshness_status')}; "
            f"quality={item.get('data_quality_status')}; "
            f"reason={item.get('reason', '')}"
        )
    print("結果を", STATE_FILE, "に保存しました。")
    print("=====================================")


if __name__ == "__main__":
    main()
