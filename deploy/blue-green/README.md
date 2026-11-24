# Blue-green deployment

Two copies of the API run side by side, each from its own version-tagged image. A router in
front sends all traffic to one of them, the live colour. A new version goes to the idle
colour, gets checked there while users keep using the live one, and then all traffic moves
to it in one step. The old version keeps running, so going back is the same one step.

```
                        +--> blue   :8001   crop-yield:v2   (live)
client --> router :8080 |
                        +--> green  :8002   crop-yield:v3   (idle: the new version)
```

| Part | File |
|---|---|
| Containers and image tags | `docker-compose.yml`, `.env` (`BLUE_TAG`, `GREEN_TAG`, `ROUTER_TAG`) |
| Router: sends each request to the colour named in `state/active`, adds the `X-Deploy-Color` header | `src/router.py` |
| Switch: checks the target colour, switches, confirms, records it | `src/blue_green.py` |
| Which colour is live, and every switch so far | `state/active`, `state/history.csv` (created on first use) |

The switch refuses to move traffic unless the target colour passes `/health`, answers a real
`/predict` request (Faisalabad wheat 2025) with a sensible yield, and then really answers
through the router. If any check fails, nothing changes.

## Steps

Run everything from the project folder, with Docker Desktop running.

**1. Build two versions as tagged images.** Each tag is a version you can always return to.
Example: v2 = `main` before blue-green (commit `aae08d0`), v3 = current `main`.

```bash
git checkout aae08d0 && docker build -t crop-yield:v2 .
git checkout main    && docker build -t crop-yield:v3 .
docker images crop-yield
```

**2. Start both colours and the router.** Blue is live at first.

```bash
docker compose -f deploy/blue-green/docker-compose.yml up -d
python -m src.blue_green status
curl -i http://localhost:8080/health      # header X-Deploy-Color: blue
```

**3. Switch to green (the new version).** It is checked first; users see no downtime.

```bash
python -m src.blue_green switch green
curl -i http://localhost:8080/health      # X-Deploy-Color: green
curl http://localhost:8080/version        # deploy_color, model version, data version, code commit
```

**4. Roll back** (if the new version misbehaves): one step, blue is still running.

```bash
python -m src.blue_green rollback
```

**5. Next release.** Build the new tag, put it on the idle colour, restart only that colour,
then switch:

```bash
docker build -t crop-yield:v4 .
# in deploy/blue-green/.env set the idle colour's tag, e.g. BLUE_TAG=v4
docker compose -f deploy/blue-green/docker-compose.yml up -d blue
python -m src.blue_green switch blue
```

**Stop:** `docker compose -f deploy/blue-green/docker-compose.yml down`

## Tested

Without Docker (the three parts as plain processes on ports 8001, 8002 and 8080), under a
steady stream of `/predict` requests through the router during a switch to green and a
rollback to blue: 1,197 requests, 0 errors. Switching to a stopped green was refused, and
blue stayed live. `tests/test_blue_green.py` checks the router and the switch rules.

## Render

The public service on Render stays a single service. Render already deploys a new version
next to the old one and moves traffic only after `/health` passes, and can roll back to an
older deploy (see `RUNBOOK.md`). This folder adds the explicit blue-green form, with both
versions running at once and a switch we control, on any machine with Docker.
