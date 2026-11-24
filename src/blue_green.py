"""Blue-green switch: check the idle colour, then move all traffic to it in one step.

Commands (run from the project folder while deploy/blue-green is up):
  python -m src.blue_green status            which colour is live, health and version of both
  python -m src.blue_green switch green      check green, then send all traffic to it
  python -m src.blue_green rollback          send traffic back to the other colour (instant)

Before switching, the target colour must pass three checks, or nothing changes:
  1. GET /health answers {"status": "ok"}
  2. POST /predict with a real request (Faisalabad wheat 2025) answers with a sensible yield
  3. after the switch, a request through the router is answered by the new colour
     (X-Deploy-Color header); if not, the old colour is put back.
Every switch is added to deploy/blue-green/state/history.csv (time, from, to, data version).
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import httpx

STATE_DIR = os.getenv("BLUE_GREEN_STATE", "deploy/blue-green/state")
ACTIVE_FILE = os.path.join(STATE_DIR, "active")
HISTORY_FILE = os.path.join(STATE_DIR, "history.csv")
URLS = {
    "router": os.getenv("ROUTER_URL", "http://localhost:8080"),
    "blue": os.getenv("BLUE_PUBLIC_URL", "http://localhost:8001"),
    "green": os.getenv("GREEN_PUBLIC_URL", "http://localhost:8002"),
}
# a real request: Faisalabad wheat 2025 (the same as in RUNBOOK.md)
SAMPLE = {"district": "Faisalabad", "province": "Punjab", "crop": "wheat", "year": 2025,
          "season_tmax_mean": 29.57, "season_rain_mm": 63.93, "heat_days_40c": 22, "rain_anomaly": -19.59,
          "soil_ph": 8.02, "soil_oc": 7.18, "soil_clay": 20.99, "soil_sand": 34.26, "soil_nitrogen": 0.73,
          "prev_yield": 3.764, "prev2_yield": 3.659, "prev3_yield": 3.276}
WHEAT_RANGE = (0.5, 8.0)  # t/ha: a prediction outside this means the model is broken


def read_active(path=ACTIVE_FILE):
    try:
        with open(path) as f:
            color = f.read().strip()
    except OSError:
        return "blue"
    return color if color in ("blue", "green") else "blue"


def write_active(color, path=ACTIVE_FILE):
    """Write in one step (temporary file + rename), so the router never reads half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(color + "\n")
    os.replace(tmp, path)


def check(color, client, urls=URLS):
    """Problems found on one colour (empty list = ready to take traffic)."""
    base = urls[color]
    try:
        r = client.get(f"{base}/health")
        if r.status_code != 200 or r.json().get("status") != "ok":
            return [f"/health answered {r.status_code} {r.text[:100]}"]
        r = client.post(f"{base}/predict", json=SAMPLE)
        if r.status_code != 200:
            return [f"/predict answered {r.status_code} {r.text[:100]}"]
        y = r.json().get("predicted_yield_t_ha")
        if y is None or not WHEAT_RANGE[0] <= y <= WHEAT_RANGE[1]:
            return [f"/predict gave {y} t/ha for Faisalabad wheat, outside {WHEAT_RANGE}"]
    except httpx.HTTPError as e:
        return [f"not reachable at {base}: {e}"]
    return []


def version_of(color, client, urls=URLS):
    try:
        r = client.get(f"{urls[color]}/version")
        return r.json() if r.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return {}


def served_by(client, urls=URLS, tries=5):
    """Which colour answers through the router (from the X-Deploy-Color header)."""
    for _ in range(tries):
        try:
            color = client.get(f"{urls['router']}/health").headers.get("X-Deploy-Color")
            if color:
                return color
        except httpx.HTTPError:
            pass
        time.sleep(1)
    return None


def switch(target, client, urls=URLS, active_file=ACTIVE_FILE, history_file=HISTORY_FILE, verify_tries=5):
    """Check `target`, switch to it, confirm through the router. Returns (ok, message)."""
    current = read_active(active_file)
    if target == current:
        return True, f"{target} is already live; nothing to do"
    problems = check(target, client, urls)
    if problems:
        return False, f"NOT switched: {target} failed its checks ({'; '.join(problems)}). {current} stays live."
    write_active(target, active_file)
    answered = served_by(client, urls, verify_tries)
    if answered != target:
        write_active(current, active_file)
        return False, f"NOT switched: the router still answered from {answered}; {current} put back."
    info = version_of(target, client, urls)
    new_file = not os.path.exists(history_file)
    with open(history_file, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time_utc", "from", "to", "model_version", "data_version", "git_commit"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), current, target,
                    info.get("registered_version"), info.get("data_version"), info.get("git_commit")])
    return True, f"Switched: {current} -> {target}. All traffic now goes to {target}; {current} is still running."


def status(client, urls=URLS, active_file=ACTIVE_FILE):
    lines = [f"Live colour: {read_active(active_file)} (router at {urls['router']})"]
    for color in ("blue", "green"):
        problems = check(color, client, urls)
        info = version_of(color, client, urls)
        lines.append(f"  {color:5s} {'ready' if not problems else 'NOT ready: ' + '; '.join(problems)}"
                     f"  model v{info.get('registered_version')}  data {info.get('data_version')}"
                     f"  code {info.get('git_commit')}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Blue-green switch for the crop yield API")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sw = sub.add_parser("switch")
    sw.add_argument("color", choices=["blue", "green"])
    sub.add_parser("rollback")
    args = ap.parse_args(argv)
    with httpx.Client(timeout=60) as client:
        if args.command == "status":
            print(status(client))
            return 0
        target = args.color if args.command == "switch" else ("green" if read_active() == "blue" else "blue")
        ok, message = switch(target, client)
        print(message)
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
