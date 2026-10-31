"""スキャンステータス記録（scan_status.json）に関するテスト。

第1段階の変更（2026-10-31）の検証用。ネットワークは使わない。

確認すること:
- build_status_record が既存の集計値（pair_results / forward_summary /
  run_health）を、ホワイトリストのキーだけに限定して転記すること。
- validation が空・欠落したペアでも壊れないこと。
- 転記対象外の情報（価格の詳細や任意の追加キー）が記録に混入しないこと。
- GitHub Actions の実行メタ情報は環境変数がある場合のみ入ること。
- write_status_file で書いた JSON が読み戻せること。
- write_status_safely は書き出しに失敗しても例外を出さず False を
  返し、成功時は True を返すこと（スキャン本体へ影響しない担保）。
"""

import json

import scanner


def _pair_results():
    return [
        {
            "name": "USD/JPY",
            "ticker": "JPY=X",
            "decision": "検証条件未達・保留",
            "direction": "買い",
            "validation": {
                "passed": False,
                "reason": "モデル検証ゲート不合格",
                "model_validation_ok": False,
                "model_gate_reasons": ["balanced_accuracy不足(0.360<0.400)"],
                "pnl_stability_ok": False,
                "pnl_gate_reasons": ["コスト込み損益がマイナス"],
                "positive_pnl_folds": 1,
                "total_pnl_trades": 42,
                "balanced_accuracy": 0.36,
                "baseline_balanced_accuracy": 0.34,
                "class_recall_pct": {"buy": 30.0, "sell": 42.0, "neutral": 50.0},
                # 転記対象外のキー（混入しないことを確認する）。
                "fold_details": [{"fold": 1, "detail": "do-not-copy"}],
                "simulated_pnl": {"net_pips": -100.43},
            },
            "data_freshness_status": "fresh",
            "data_age_minutes": 12.5,
            "data_quality_status": "ok",
            "news_status": "recent_articles_no_risk_keywords",
            # 転記対象外のキー。
            "price": 157.5,
            "atr": 0.42,
        },
        {
            "name": "EUR/USD",
            "ticker": "EURUSD=X",
            "decision": "判定保留",
            "direction": None,
            # validation 自体が無いペア（検証スキップ時）。
            "data_freshness_status": "stale_on_open_day",
            "data_age_minutes": 400.0,
            "data_quality_status": "ok",
            "news_status": "unknown",
        },
    ]


def _forward_summary():
    return {
        "tracking_since": "2026-10-26T01:14:47+09:00",
        "evaluated_total": 0,
        "hit": 0,
        "miss": 0,
        "not_reached": 0,
        "hit_rate_pct": None,
        "buy_hit_rate_pct": None,
        "sell_hit_rate_pct": None,
        "net_pips_total": 0.0,
        "avg_mae_pips": None,
        "pending_count": 0,
        "stored_predictions": 0,
        "evaluated_this_run": 0,
        "recorded_this_run": 0,
        "recorded_verified_this_run": 0,
        "recorded_unverified_this_run": 0,
        "news_verified_breakdown": {},
        "definitions": "的中率=HIT/(HIT+MISS)。",
        # 転記対象外のキー。
        "internal_note": "do-not-copy",
    }


def _run_health():
    return {
        "pairs_total": 2,
        "fresh_data_pairs": 1,
        "stale_or_unknown_pairs": 1,
        "data_quality_fail_pairs": 0,
        "validation_pass_pairs": 0,
        "validation_skipped_or_not_passed_pairs": 2,
        "news_unavailable_pairs": 0,
        "news_retrieval_error_pairs": 0,
        "news_empty_result_pairs": 0,
        "news_no_recent_article_pairs": 0,
        "news_recent_articles_checked_pairs": 2,
        "candidate_pairs": 0,
        "execution_note": "note",
        # 転記対象外のキー（混入しないことを確認する）。
        "legacy_extra": "do-not-copy",
    }


def test_build_status_record_copies_whitelisted_values():
    record = scanner.build_status_record(
        "2026-10-31T12:00:00+09:00",
        "2026-10-31T12:03:00+09:00",
        _pair_results(),
        _forward_summary(),
        _run_health(),
    )

    assert record["schema_version"] == scanner.STATUS_SCHEMA_VERSION
    assert record["started_at"] == "2026-10-31T12:00:00+09:00"
    assert record["finished_at"] == "2026-10-31T12:03:00+09:00"
    assert record["generated_at_jst"] == "2026-10-31T12:03:00+09:00"

    pair = record["pairs"][0]
    assert pair["name"] == "USD/JPY"
    assert pair["decision"] == "検証条件未達・保留"
    assert pair["validation"]["passed"] is False
    assert pair["validation"]["balanced_accuracy"] == 0.36
    assert pair["validation"]["model_gate_reasons"] == [
        "balanced_accuracy不足(0.360<0.400)"
    ]
    assert pair["data_freshness_status"] == "fresh"
    assert pair["data_age_minutes"] == 12.5

    forward = record["forward"]
    assert forward["tracking_since"] == "2026-10-26T01:14:47+09:00"
    assert forward["pending_count"] == 0
    assert forward["evaluated_total"] == 0

    assert record["run_health"]["pairs_total"] == 2


def test_build_status_record_excludes_non_whitelisted_keys():
    record = scanner.build_status_record(
        "2026-10-31T12:00:00+09:00",
        "2026-10-31T12:03:00+09:00",
        _pair_results(),
        _forward_summary(),
        _run_health(),
    )

    pair = record["pairs"][0]
    assert "price" not in pair
    assert "atr" not in pair
    assert "fold_details" not in pair["validation"]
    assert "simulated_pnl" not in pair["validation"]
    assert "internal_note" not in record["forward"]

    dumped = json.dumps(record, ensure_ascii=False)
    assert "do-not-copy" not in dumped


def test_build_status_record_run_health_whitelist():
    record = scanner.build_status_record(
        "2026-10-31T12:00:00+09:00",
        "2026-10-31T12:03:00+09:00",
        [],
        {},
        {"pairs_total": 5, "unexpected_key": "do-not-copy"},
    )

    run_health = record["run_health"]
    # 転記されるのはホワイトリストのキーだけ。
    assert set(run_health.keys()) == set(scanner._STATUS_RUN_HEALTH_KEYS)
    assert "unexpected_key" not in run_health
    # 存在する値はそのまま転記され、欠けたキーは None で補われる。
    assert run_health["pairs_total"] == 5
    assert run_health["candidate_pairs"] is None
    assert run_health["execution_note"] is None

    dumped = json.dumps(record, ensure_ascii=False)
    assert "do-not-copy" not in dumped


def test_build_status_record_handles_missing_validation():
    record = scanner.build_status_record(
        "2026-10-31T12:00:00+09:00",
        "2026-10-31T12:03:00+09:00",
        _pair_results(),
        _forward_summary(),
        _run_health(),
    )

    pair = record["pairs"][1]
    assert pair["name"] == "EUR/USD"
    assert pair["data_freshness_status"] == "stale_on_open_day"
    for key in scanner._STATUS_VALIDATION_KEYS:
        assert key in pair["validation"]
        assert pair["validation"][key] is None


def test_build_status_record_run_metadata(monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_NUMBER", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    record = scanner.build_status_record(
        "2026-10-31T12:00:00+09:00",
        "2026-10-31T12:03:00+09:00",
        [],
        {},
        {},
    )
    assert record["run"] == {
        "run_id": None,
        "run_number": None,
        "commit_sha": None,
    }
    assert record["pairs"] == []

    monkeypatch.setenv("GITHUB_RUN_ID", "123456789")
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "95")
    monkeypatch.setenv("GITHUB_SHA", "4190ffbc")
    record = scanner.build_status_record(
        "2026-10-31T12:00:00+09:00",
        "2026-10-31T12:03:00+09:00",
        [],
        {},
        {},
    )
    assert record["run"]["run_id"] == "123456789"
    assert record["run"]["run_number"] == 95
    assert record["run"]["commit_sha"] == "4190ffbc"


def test_write_status_file_roundtrip(tmp_path):
    record = scanner.build_status_record(
        "2026-10-31T12:00:00+09:00",
        "2026-10-31T12:03:00+09:00",
        _pair_results(),
        _forward_summary(),
        _run_health(),
    )
    target = tmp_path / "scan_status.json"
    scanner.write_status_file(record, path=str(target))

    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == record


def test_write_status_safely_success_and_failure(tmp_path, monkeypatch):
    written = {}

    def fake_write(record, path=scanner.STATUS_FILE):
        written["record"] = record
        return str(tmp_path / "scan_status.json")

    monkeypatch.setattr(scanner, "write_status_file", fake_write)
    ok = scanner.write_status_safely(
        "2026-10-31T12:00:00+09:00",
        _pair_results(),
        _forward_summary(),
        _run_health(),
    )
    assert ok is True
    assert written["record"]["pairs"][0]["name"] == "USD/JPY"

    def failing_write(record, path=scanner.STATUS_FILE):
        raise OSError("disk full")

    monkeypatch.setattr(scanner, "write_status_file", failing_write)
    ok = scanner.write_status_safely(
        "2026-10-31T12:00:00+09:00",
        _pair_results(),
        _forward_summary(),
        _run_health(),
    )
    assert ok is False
