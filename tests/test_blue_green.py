"""Tests for the blue-green router (src/router.py) and switch (src/blue_green.py).

The two colours are stand-ins (httpx MockTransport) that answer like the real API; the
end-to-end run with real containers is in deploy/blue-green/README.md.
"""

import httpx
from fastapi.testclient import TestClient

from src import blue_green as bg
from src import router

URLS = {"router": "http://router", "blue": "http://blue", "green": "http://green"}


def fake_api(broken=(), state=None):
    """Answers for blue and green; a colour in `broken` fails /predict. The router URL
    answers from whichever colour is in the active file."""
    def handler(request):
        host = request.url.host
        if host == "router":
            host = bg.read_active(state)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"}, headers={"X-Deploy-Color": host})
        if request.url.path == "/predict":
            if host in broken:
                return httpx.Response(503, text="model file missing")
            return httpx.Response(200, json={"predicted_yield_t_ha": 3.65})
        if request.url.path == "/version":
            return httpx.Response(200, json={"registered_version": 2 if host == "green" else 1,
                                             "data_version": "a1b9813dc42f", "git_commit": "abc123"})
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------- router ----------

def test_router_sends_traffic_to_the_colour_in_the_file(tmp_path, monkeypatch):
    active = tmp_path / "active"
    monkeypatch.setattr(router, "ACTIVE_FILE", str(active))
    seen = []

    def upstream(request):
        seen.append(request.url.host)
        return httpx.Response(200, json={"status": "ok"})

    router.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    monkeypatch.setattr(router, "UPSTREAMS", {"blue": "http://blue", "green": "http://green"})
    client = TestClient(router.app)
    try:
        active.write_text("blue\n")
        assert client.get("/health").headers["X-Deploy-Color"] == "blue"
        active.write_text("green\n")  # the switch: no restart
        r = client.get("/health")
        assert r.headers["X-Deploy-Color"] == "green" and r.json() == {"status": "ok"}
        assert seen == ["blue", "green"]
        assert client.get("/router/status").json()["active"] == "green"
    finally:
        router.app.state.client = None


def test_router_defaults_to_blue_without_a_valid_file(tmp_path):
    assert router.active_color(str(tmp_path / "missing")) == "blue"
    (tmp_path / "bad").write_text("purple")
    assert router.active_color(str(tmp_path / "bad")) == "blue"


# ---------- switch ----------

def test_switch_moves_traffic_after_the_checks_pass(tmp_path):
    active, history = str(tmp_path / "active"), str(tmp_path / "history.csv")
    bg.write_active("blue", active)
    ok, msg = bg.switch("green", fake_api(state=active), URLS, active, history, verify_tries=1)
    assert ok and bg.read_active(active) == "green"
    lines = open(history).read().splitlines()
    assert lines[0].startswith("time_utc,from,to") and ",blue,green,2,a1b9813dc42f," in lines[1]


def test_a_broken_colour_never_gets_traffic(tmp_path):
    active, history = str(tmp_path / "active"), str(tmp_path / "history.csv")
    bg.write_active("blue", active)
    ok, msg = bg.switch("green", fake_api(broken={"green"}, state=active), URLS, active, history, verify_tries=1)
    assert not ok and "NOT switched" in msg and bg.read_active(active) == "blue"


def test_rollback_is_switching_back(tmp_path):
    active, history = str(tmp_path / "active"), str(tmp_path / "history.csv")
    bg.write_active("green", active)
    ok, _ = bg.switch("blue", fake_api(state=active), URLS, active, history, verify_tries=1)
    assert ok and bg.read_active(active) == "blue"


def test_switch_to_the_live_colour_does_nothing(tmp_path):
    active = str(tmp_path / "active")
    bg.write_active("blue", active)
    ok, msg = bg.switch("blue", fake_api(state=active), URLS, active, str(tmp_path / "h.csv"))
    assert ok and "already live" in msg
