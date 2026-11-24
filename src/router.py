"""Blue-green router: one public address in front of two copies of the API.

  client --> router (port 8080) --> blue  (one image tag, e.g. crop-yield:v2)
                                \\-> green (another tag, e.g. crop-yield:v3)

Every request goes to the ACTIVE colour, named in a small text file (ACTIVE_FILE, "blue" or
"green"). Switching is writing the other colour into that file (src/blue_green.py does it, after
checking the new colour works). The router reads the file on every request, so a switch takes
effect at once, without restarting anything and without dropping requests; rolling back is
switching to the old colour again, which is still running.

Every response carries X-Deploy-Color, so you can see which colour answered.
GET /router/status shows the active colour and the address of each colour.

Run (inside deploy/blue-green/docker-compose.yml):  uvicorn src.router:app --host 0.0.0.0 --port 8000
"""

import os

import httpx
from fastapi import FastAPI, Request, Response

COLORS = ("blue", "green")
ACTIVE_FILE = os.getenv("ACTIVE_FILE", "deploy/blue-green/state/active")
UPSTREAMS = {
    "blue": os.getenv("BLUE_URL", "http://localhost:8001"),
    "green": os.getenv("GREEN_URL", "http://localhost:8002"),
}
DEFAULT_COLOR = "blue"
# headers that describe one connection, not the content: not passed on
# (date and server too: the router's own server adds them, so they would appear twice)
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
               "proxy-authorization", "proxy-authenticate", "host", "content-length", "content-encoding",
               "date", "server"}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)  # /docs etc. come from the active colour
app.state.client = None  # created on first use; tests put a client with a fake transport here


def active_color(path=None):
    """The colour in ACTIVE_FILE; blue if the file is missing or holds anything else."""
    try:
        with open(path or ACTIVE_FILE) as f:
            color = f.read().strip()
    except OSError:
        return DEFAULT_COLOR
    return color if color in COLORS else DEFAULT_COLOR


def _client():
    if app.state.client is None:
        app.state.client = httpx.AsyncClient(timeout=60)
    return app.state.client


@app.get("/router/status")
def status():
    return {"active": active_color(), "upstreams": UPSTREAMS, "active_file": ACTIVE_FILE}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request):
    color = active_color()
    url = f"{UPSTREAMS[color]}/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
    try:
        upstream = await _client().request(request.method, url, params=request.query_params,
                                           content=await request.body(), headers=headers)
    except httpx.HTTPError as e:
        return Response(f"{color} is not reachable: {e}", status_code=502, headers={"X-Deploy-Color": color})
    out = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS}
    out["X-Deploy-Color"] = color
    return Response(upstream.content, status_code=upstream.status_code, headers=out)
