"""forward_recordable の記録経路を固定する回帰テスト。

ニュース確認だけを理由に保留された予測は、売買候補とは分離した
未確認状態（news_verified=False）で forward 評価用に記録できる。
ニュース以外のゲートで保留された予測は引き続き記録しない。
ネットワーク・実データ・モデル学習は使わない。
"""

import scanner


def make_news_only_hold(**overrides):
    item = {
        "name": "米ドル/円 (USD/JPY)",
        "ticker": "USDJPY=X",
        "decision": "検証条件未達・保留",
        "direction": "売り",
        "price": 150.123,
        "atr": 0.75,
        "signal_bar_time": "2026-01-05T00:00:00+00:00",
        "news_status": "重要ニュースあり",
        "forward_recordable": True,
        "feature_snapshot": {"Return_1": 0.001},
    }
    item.update(overrides)
    return item


def test_news_only_hold_is_recorded_as_unverified():
    predictions = []
    details = {}

    count = scanner.record_forward_predictions(
        predictions, [make_news_only_hold()], details=details
    )

    assert count == 1
    assert len(predictions) == 1
    recorded = predictions[0]
    assert recorded["status"] == "PENDING"
    assert recorded["news_verified"] is False
    assert recorded["news_status_at_record"] == "重要ニュースあり"
    assert recorded["direction"] == "売り"
    assert recorded["direction_sign"] == -1
    assert recorded["reference_price"] == 150.123
    assert recorded["atr"] == 0.75
    assert recorded["pip_size"] == 0.01
    assert recorded["spread_pips"] == 0.2
    assert recorded["feature_snapshot"] == {"Return_1": 0.001}
    assert details == {"recorded_verified": 0, "recorded_unverified": 1}


def test_news_only_hold_dedupes_by_ticker_and_signal_time():
    predictions = []
    item = make_news_only_hold()

    assert scanner.record_forward_predictions(predictions, [item]) == 1
    assert scanner.record_forward_predictions(predictions, [item]) == 0
    assert len(predictions) == 1


def test_non_news_gate_hold_is_not_recorded():
    predictions = []
    item = make_news_only_hold(forward_recordable=False)

    assert scanner.record_forward_predictions(predictions, [item]) == 0
    assert predictions == []


def test_forward_recordable_item_with_invalid_atr_is_not_recorded():
    predictions = []
    item = make_news_only_hold(atr=0)

    assert scanner.record_forward_predictions(predictions, [item]) == 0
    assert predictions == []
