"""Round 2 (25 Nov 2026): every endpoint answers garbage with a 4xx, never a 500."""
import re

import pytest
from fastapi.testclient import TestClient

from conftest import register

GARBAGE_BODIES = [None, {}, [], "x", 0, -1, 1e308, {"a": "b"}, {"strategy": "x"}, {"spec": []}, {"spec": {}, "instruction": 5},
                  {"text": None}, {"strategy_id": "abc"}, {"days": "lots"}, {"email": 1, "password": []}, {"x": "\u0000" * 10},
                  {"strategy": {"entry": {"long": [{"left": "close", "op": "gt", "right": "nan"}]}}}, {"strategy_id": 10**20},
                  {"broker": "nope", "credentials": "x"}, {"spec": {"x": 1}, "instruction": "exit at 3"}, {"seed": -1, "text": "buy nifty at 9:20"},
                  {"now": "not-a-time"}, {"plan": "platinum"}, {"strategy_id": 1, "mode": "live", "confirm_text": "LIVE"}]
PATH_VALUES = ["1", "0", "-1", "abc", "99999999999999999999"]
INTENTIONAL_5XX = {502, 503}  # broker unreachable / payments not configured are deliberate


@pytest.fixture()
def fuzz_client(client):
    from app.config import get_settings

    get_settings().rate_limit_enabled = False
    yield TestClient(client.app, raise_server_exceptions=False)
    get_settings().rate_limit_enabled = True


def test_no_endpoint_returns_500_on_garbage(client, fuzz_client):
    user = register(client)
    admin = register(client, "admin@devcognix.com")
    client.post("/api/strategies", json={"prompt": "Buy nifty ATM call when 9 EMA crosses above 21 EMA on 5 min"}, headers=user)
    failures = []
    sent = 0
    # FastAPI >= 0.16x nests included routers, so enumerate endpoints from the OpenAPI schema
    # (plus the hidden ones) instead of app.routes.
    endpoints = [(m.upper(), p) for p, ops in client.app.openapi()["paths"].items() for m in ops]
    endpoints += [("POST", "/api/billing/webhook")]
    assert len(endpoints) > 50
    for method, route_path in endpoints:
        if True:
            for pv in PATH_VALUES if "{" in route_path else ["-"]:
                path = re.sub(r"\{[^}]+\}", pv, route_path)
                for hdr in ({}, user, admin, {"Authorization": "Bearer garbage"}):
                    bodies = GARBAGE_BODIES if method in ("POST", "PUT", "PATCH") else [None]
                    for body in bodies:
                        kw = {"headers": dict(hdr)}
                        if isinstance(body, str):
                            kw["content"] = body.encode()
                            kw["headers"]["content-type"] = "application/json"
                        elif body is not None:
                            kw["json"] = body
                        if method == "GET":
                            kw["params"] = {"limit": "x", "offset": "-5", "date_from": "nope", "session_id": "abc"}
                        r = fuzz_client.request(method, path, **kw)
                        sent += 1
                        if r.status_code >= 500 and r.status_code not in INTENTIONAL_5XX:
                            failures.append((method, path, str(body)[:60], r.status_code))
    assert sent > 2000
    assert not failures, sorted({(m, p, c) for m, p, _b, c in failures})


def test_invalid_spec_422_is_json_serialisable(client):
    # model_validator errors carry the raw ValueError in ctx; that used to crash the 422 into a 500
    r = client.post("/api/strategies/explain", json={"entry": {"long": []}})
    assert r.status_code == 422 and r.json()["detail"]["message"] == "Invalid strategy"


def test_negative_seed_is_422_not_500(client):
    r = client.post("/api/backtests/run", json={"text": "Buy nifty ATM call at 9:20", "seed": -1})
    assert r.status_code == 422


def test_paper_account_backtest_is_422_not_502(client):
    h = register(client)
    acc = client.post("/api/brokers/accounts", json={"broker": "paper", "credentials": {}, "test": False}, headers=h)
    assert acc.status_code == 200, acc.text
    r = client.post("/api/backtests/run", json={"text": "Buy nifty ATM call at 9:20", "days": 5, "broker_account_id": acc.json()["account"]["id"]}, headers=h)
    assert r.status_code == 422 and "synthetic" in r.json()["detail"]


def test_websocket_subprotocol_auth_contract(client):
    """Contract §3: client connects with subprotocols ["tc-auth", <jwt>]; server echoes tc-auth."""
    from starlette.websockets import WebSocketDisconnect

    token = register(client)["Authorization"].split()[1]
    with client.websocket_connect("/ws/live", subprotocols=["tc-auth", token]) as ws:
        assert ws.accepted_subprotocol == "tc-auth"
        assert ws.receive_json()["type"] == "hello"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/live", subprotocols=["tc-auth", "not-a-jwt"]) as ws:
            ws.receive_json()


def test_out_of_range_ids_are_422(client):
    h = register(client)
    huge = 10**20
    for method, url in [("get", f"/api/strategies/{huge}"), ("delete", f"/api/strategies/{huge}"), ("get", f"/api/backtests/{huge}"),
                        ("get", f"/api/brokers/accounts/{huge}/funds"), ("get", "/api/engine/orders?session_id=99999999999999999999")]:
        assert client.request(method, url, headers=h).status_code == 422, url
    assert client.post("/api/backtests/run", json={"strategy_id": huge}, headers=h).status_code == 422
    assert client.get("/api/strategies/12345", headers=h).status_code == 404  # in range, just missing


def test_engine_refuses_futures_sessions_up_front(client):
    h = register(client)
    s = client.post("/api/strategies", json={"prompt": "Buy nifty futures when 9 EMA crosses above 21 EMA on 5 min"}, headers=h).json()
    assert s["spec"]["instrument_type"] == "future"
    r = client.post("/api/engine/start", json={"strategy_id": s["id"]}, headers=h)
    assert r.status_code == 400 and "futures" in r.json()["detail"]
