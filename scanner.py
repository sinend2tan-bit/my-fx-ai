import json
import os
import re
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
MIN_TRAIN_ROWS = 180
N_WALK_FORWARD_FOLDS = 3
FAST_MOVE_ATR_MULTIPLE = 2.5

# ニュースを新しいものとみなす時間
NEWS_MAX_AGE_HOURS = 12

# 重要ニュースを検出した場合は新しい売買候補を保留
NEWS_HOLD_ON_IMPORTANT = True

# ニュースが確認できない場合も新しい売買候補を保留
NEWS_HOLD_IF_UNAVAILABLE = True

MODEL_VERSION = 8

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


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state):
    temp_file = STATE_FILE + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )

    os.replace(temp_file, STATE_FILE)


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



def fetch_news(ticker):
    """
    Yahoo Financeからニュースを取得する。
    複数の取得方法を順番に試し、取得失敗を
    「重要ニュースなし」と誤認しない。
    """
    result = {
        "news_checked_at_jst": iso_now(),
        "news_status": "unknown",
        "news_article_count": 0,
        "news_recent_count": 0,
        "news_risk_detected": False,
        "news_risk_reason": "",
        "news_risk_headlines": [],
        "news_headlines": [],
        "news_source": "Yahoo Finance",
        "news_retrieval_method": None,
        "news_retrieval_errors": [],
    }

    raw_news = []
    retrieval_errors = []

    # 方法1：yf.Search が利用可能なら試す
    search_api = getattr(yf, "Search", None)

    if callable(search_api):
        try:
            search_result = search_api(
                ticker, news_count=10
            )
            candidate = getattr(
                search_result, "news", None
            )

            if isinstance(candidate, list) and candidate:
                raw_news = candidate
                result["news_retrieval_method"] = "yf.Search"

        except Exception as exc:
            retrieval_errors.append(
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

                method = "Ticker.get_news"
            else:
                candidate = ticker_obj.news
                method = "Ticker.news"

            if isinstance(candidate, list) and candidate:
                raw_news = candidate
                result["news_retrieval_method"] = method

        except Exception as exc:
            retrieval_errors.append(
                "Ticker news: "
                + type(exc).__name__
                + ": "
                + str(exc)[:120]
            )

    result["news_retrieval_errors"] = retrieval_errors[:3]

    if not raw_news:
        result["news_status"] = (
            "error" if retrieval_errors else "empty"
        )

        if retrieval_errors:
            result["news_risk_reason"] = (
                "複数のニュース取得方法でエラーが発生しました。"
                "重要ニュースがないとは判断できません。"
            )
        else:
            result["news_risk_reason"] = (
                "ニュース記事が返されませんでした。"
                "重要ニュースがないとは判断できません。"
            )

        print(
            f"[NEWS FETCH] ticker={ticker}, "
            f"status={result['news_status']}, "
            f"method=None, "
            f"errors={len(retrieval_errors)}"
        )
        return result

    result["news_article_count"] = len(raw_news)

    current_time = now_jst()
    cutoff = current_time - timedelta(
        hours=NEWS_MAX_AGE_HOURS
    )

    recent_articles = []

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

        published = item.pop("_published")
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

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
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

    if len(raw) < 221:
        print(f"[DATA SHORT] {ticker} {interval}: {len(raw)} bars")
        return None

    # 最新の取得足は進行中の可能性があるため除外する。
    raw = raw.iloc[:-1].copy()

    if interval == "4h":
        raw = raw.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna(subset=required)

    if len(raw) < 220:
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
            f"std={vol_std}"
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
        f"time={latest_index.isoformat()}, "
        f"close={latest_data['Close'].iloc[-1]}"
    )

    close_values = feature_df["Close"].to_numpy(dtype=float)
    high_values = feature_df["High"].to_numpy(dtype=float)
    low_values = feature_df["Low"].to_numpy(dtype=float)
    atr_values = feature_df["ATR"].to_numpy(dtype=float)

    targets = np.full(len(feature_df), np.nan)

    for i in range(len(feature_df) - LOOKAHEAD_BARS):
        price = close_values[i]
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
    # 予測時点の終値から8本後の終値までを仮想保有した場合を計算する。
    training_df["Future_Close"] = feature_df["Close"].shift(-LOOKAHEAD_BARS)
    training_df = training_df.dropna(
        subset=FEATURE_COLUMNS + ["Target", "ATR", "Close", "Future_Close"]
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


    print(
        f"[TRAINING DATA] {ticker} {interval}: "
        f"rows={len(training_df)}, "
        f"last_labeled={training_df.index[-1].isoformat()}, "
        f"latest_confirmed={latest_index.isoformat()}, "
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
        """非重複・検証区間内の概算。実約定成績ではない。"""
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

            entry = safe_float(rows.at[idx, "Close"])
            exit_price = safe_float(rows.at[idx, "Future_Close"])
            if entry is None or exit_price is None:
                continue

            gross_pips = (exit_price - entry) * direction / pip_size
            net_pips = gross_pips - float(spread_pips)
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
                "net_pips": float(net_pips),
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
    model_validation_ok = (
        score >= 0.40
        and score >= baseline_score
        and recall_pct["buy"] >= 20.0
        and recall_pct["sell"] >= 20.0
    )
    passed = bool(model_validation_ok and pnl_stability_ok)

    if not model_validation_ok:
        reason = "複数期間の時系列検証条件を満たさない"
    elif not pnl_stability_ok:
        reason = "仮想損益の期間別安定性条件を満たさない（利益保証ではありません）"
    else:
        reason = "時系列検証と仮想損益の期間別安定性条件を満たしました（実取引成績ではありません）"

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
        f"spread_cost_pips={spread_all}, net_pips={total_pnl['net_pips']}, "
        f"note=spread deducted once per simulated trade; excludes slippage/swaps/real fills"
    )

    print(
        f"[SIMULATED PNL SUMMARY] pair={pair_name}, "
        f"trades={total_pnl['trades']}, "
        f"net_pips={total_pnl['net_pips']}, "
        f"win_rate={total_pnl['win_rate_pct']}, "
        f"avg_pips={total_pnl['avg_pips_per_trade']}, "
        f"max_drawdown_pips={total_pnl['max_drawdown_pips']}, "
        f"spread_pips={spread_pips}, hold_bars={LOOKAHEAD_BARS}, "
        f"note=non-overlapping close-to-close estimate; last hold window per fold excluded; not executable results"
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
        "pnl_stability_ok": bool(pnl_stability_ok),
        "positive_pnl_folds": int(positive_pnl_folds),
        "total_pnl_trades": int(total_pnl["trades"]),
        "balanced_accuracy": round(float(score), 4),
        "baseline_balanced_accuracy": round(
            float(baseline_score), 4
        ),
        "class_recall_pct": recall_pct,
        "folds": used_folds,
        "fold_details": fold_details,
        "simulated_pnl": total_pnl,
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



def analyze(df, validation):
    if df is None or len(df) < MIN_TRAIN_ROWS:
        return {
            "direction": "様子見",
            "buy_pct": None,
            "sell_pct": None,
            "neutral_pct": None,
            "reason": "学習データ不足",
        }

    if not validation.get("passed"):
        return {
            "direction": "様子見",
            "buy_pct": None,
            "sell_pct": None,
            "neutral_pct": None,
            "reason": validation.get("reason", "検証条件未達"),
        }

    latest_data = df.attrs.get("latest_data")
    if latest_data is None or latest_data.empty:
        return {
            "direction": "様子見",
            "buy_pct": None,
            "sell_pct": None,
            "neutral_pct": None,
            "reason": "最新の確定足を確認できません",
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
                "最新の確定足を使ったモデル推定値です。"
                "将来の利益を保証するものではありません。"
            ),
        }

    except Exception as exc:
        return {
            "direction": "様子見",
            "buy_pct": None,
            "sell_pct": None,
            "neutral_pct": None,
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




def data_age_minutes(df):
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

        last_time = last_time.tz_convert("Asia/Tokyo")
        now = pd.Timestamp.now(tz="Asia/Tokyo")
        age = (now - last_time).total_seconds() / 60

        print(
            f"[DATA AGE DEBUG] latest_confirmed={last_time.isoformat()}, "
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
        "validation": {},
        "buy_pct": None,
        "sell_pct": None,
        "neutral_pct": None,
        **news,
    }

    try:
        # メイン分析は1時間足。4時間足・日足で方向の偏りを確認。
        hourly = bars(ticker, interval="1h", period="60d")
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

        result["price"] = safe_float(latest_data["Close"].iloc[-1])
        result["data_age_minutes"] = data_age_minutes(hourly)

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
        })

        
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

        if age > 180:
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

        # ニュースの確認結果を候補判定に反映する。
        news_allowed, news_reason = news_allows_candidate(news)
        if not news_allowed:
            result["decision"] = "ニュース確認・判定保留"
            result["reason"] = news_reason
            return result

        # 上位足の方向と大きく食い違う候補は保留。
        direction = model_result["direction"]
        if direction == "買い" and "下降" in trends:
            result["decision"] = "上位足と不一致・保留"
            result["reason"] = "上位足に下降トレンドがあるため、新規買い候補を保留します。"
            return result

        if direction == "売り" and "上昇" in trends:
            result["decision"] = "上位足と不一致・保留"
            result["reason"] = "上位足に上昇トレンドがあるため、新規売り候補を保留します。"
            return result

        if direction == "様子見":
            result["decision"] = "様子見"
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
# 全体実行・結果保存
# =========================================================

def main():
    started_at = iso_now()
    previous_state = load_state()

    pair_results = []
    for name, config in PAIRS.items():
        pair_results.append(analyze_pair(name, config))

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

    history = previous_state.get("history", [])
    if not isinstance(history, list):
        history = []

    run_record = {
        "run_at_jst": started_at,
        "summary": summary,
        "pairs": pair_results,
    }
    history.append(run_record)
    history = history[-99:]

    news_status_counts = {}
    for item in pair_results:
        status = item.get("news_status", "unknown")
        news_status_counts[status] = news_status_counts.get(status, 0) + 1

    state = {
        "schema_version": MODEL_VERSION,
        "last_run_jst": started_at,
        "mode": "multi_period_validation_with_news_risk_gate_no_auto_orders",
        "interval": "1h",
        "news_policy": {
            "source": "Yahoo Finance",
            "max_age_hours": NEWS_MAX_AGE_HOURS,
            "hold_on_important_news": NEWS_HOLD_ON_IMPORTANT,
            "hold_if_news_unavailable": NEWS_HOLD_IF_UNAVAILABLE,
            "notice": (
                "ニュース見出しを売買方向の判断には使用しない。"
                "重要キーワード検出時やニュース確認不能時は新規候補を保留する。"
                "公式経済指標カレンダーの代替ではない。"
            ),
        },
        "news_status_counts": news_status_counts,
        "summary": summary,
        "pairs": pair_results,
        "history": history,
        "safety_notice": (
            "価格分析・注文補助専用です。証券口座接続・自動発注・自動決済は行いません。"
            "分析結果は将来の利益を保証しません。"
        ),
    }

    save_state(state)

    print("\n========== FX SCAN SUMMARY ==========")
    print("\n".join(summary))
    print("ニュース取得状況:", news_status_counts)
    print("結果を", STATE_FILE, "に保存しました。")
    print("=====================================")


if __name__ == "__main__":
    main()
