"""Regression tests for directional PnL hold propagation in analyze_pair."""

import pandas as pd

import scanner


def test_directional_pnl_failure_cannot_become_a_candidate(monkeypatch):
    """A failed BUY PnL gate must remain a hold through the final decision."""
    latest = pd.DataFrame(
        {"Close": [150.0], "ATR": [0.75]},
        index=pd.DatetimeIndex(["2026-11-18T09:00:00Z"]),
    )
    hourly = pd.DataFrame({"Close": [150.0]})
    hourly.attrs["latest_data"] = latest
    hourly.attrs["latest_history"] = latest

    monkeypatch.setattr(scanner, "fetch_news", lambda ticker: {})
    monkeypatch.setattr(scanner, "bars", lambda *args, **kwargs: hourly)
    monkeypatch.setattr(scanner, "timeframe_trend", lambda frame: "横ばい")
    monkeypatch.setattr(scanner, "validate_latest_bar", lambda frame: (True, "ok"))
    monkeypatch.setattr(scanner, "data_age_minutes", lambda frame: 10.0)
    monkeypatch.setattr(scanner, "fast_move_detected", lambda frame: (False, None))
    monkeypatch.setattr(
        scanner,
        "walk_forward_validation",
        lambda frame, pair_name=None: {
            "passed": True,
            "reason": "synthetic validation pass",
            "directional_pnl": {
                "buy": {"passed": False, "trades": 4, "net_pips": -2.0},
                "sell": {"passed": True, "trades": 12, "net_pips": 3.0},
            },
        },
    )
    monkeypatch.setattr(
        scanner,
        "analyze",
        lambda frame, validation: {
            "direction": "買い",
            "buy_pct": 70.0,
            "sell_pct": 10.0,
            "neutral_pct": 20.0,
            "feature_snapshot": {"synthetic": True},
            "reason": "synthetic model direction",
        },
    )
    monkeypatch.setattr(scanner, "news_allows_candidate", lambda news: (True, ""))

    result = scanner.analyze_pair(
        "米ドル/円 (USD/JPY)",
        {"ticker": "USDJPY=X", "spread_pips": 0.2, "pip": 0.01},
    )

    assert result["direction"] == "様子見"
    assert result["directional_pnl_hold"] is True
    assert result["decision"] == "方向別検証未達・保留"
    assert "仮想損益" in result["reason"]
    assert result["forward_recordable"] is False
    assert result["decision"] != "候補あり（手動確認のみ）"
