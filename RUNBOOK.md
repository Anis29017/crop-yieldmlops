# Runbook: Crop Yield Prediction API

How to check, redeploy and roll back the deployed service.

## The service

| | |
|---|---|
| Live URL | https://crop-yield-api-hy6r.onrender.com (opens the API docs) |
| Platform | Render, Free web service, region Singapore, service name `crop-yield-api` |
| Built from | `Dockerfile` in this repository, branch `main` |
| Deploys | Automatically when `main` changes (Render: **Settings → Auto-Deploy**) |
| Build time | About 4 minutes (dependencies + training the model inside the build) |

The free service sleeps after 15 minutes without traffic and takes about a minute to wake.
Open `/health` a few minutes before any demo or grading.

## Endpoints

| Endpoint | Use |
|---|---|
| `GET /health` | Liveness: returns `{"status": "ok"}`. Render and the Docker healthcheck use it |
| `POST /predict` | Yield prediction (t/ha) for one district, crop and year |
| `GET /stats` | Request count, average / p95 / max latency per endpoint, model prediction time (resets on restart) |
| `GET /docs` | Interactive API docs (`/` redirects here) |

## Check that the service works

Run after every deploy or rollback.

```bash
URL=https://crop-yield-api-hy6r.onrender.com

curl $URL/health

curl -X POST $URL/predict -H "Content-Type: application/json" -d '{
  "district": "Faisalabad", "province": "Punjab", "crop": "wheat", "year": 2025,
  "season_tmax_mean": 29.57, "season_rain_mm": 63.93, "heat_days_40c": 22, "rain_anomaly": -19.59,
  "soil_ph": 8.02, "soil_oc": 7.18, "soil_clay": 20.99, "soil_sand": 34.26, "soil_nitrogen": 0.73,
  "prev_yield": 3.764, "prev2_yield": 3.659, "prev3_yield": 3.276}'

curl $URL/stats
```

Expected: `{"status":"ok"}`, a `predicted_yield_t_ha` of about 3.65, and `/stats` counting the requests.
Every request is also logged with its latency: Render → **Logs**.

## Redeploy

**Normal release.** Open a pull request → CI runs lint, test and the Docker build → merge into `main` →
Render builds and deploys the new version. The old version keeps serving until the new one passes `/health`.

**Manual redeploy** (same code, e.g. after changing a setting): Render → service → **Deploys → Manual Deploy →
Deploy latest commit**. If a build behaves oddly, use **Clear build cache & deploy**.

## Roll back

### 1. On Render (fastest: the service is broken now)

1. Render → service → **Deploys** (the deploy history; older dashboards call it **Events**).
2. Find the last deploy that worked and click **Rollback**.
3. Click **Rollback to this deploy**. Render redeploys that version's image without rebuilding.
4. Run the checks above.
5. Check **Settings → Auto-Deploy**: while the broken commit is still on `main`, a new merge would deploy it again.

### 2. In the code (permanent fix)

Undo the bad change on `main`, so the next deploy is good again:

```bash
git checkout main && git pull
git revert -m 1 <merge-commit-of-the-bad-pull-request>
git push            # through a pull request if main is protected
```

Render then deploys the reverted code automatically.

### 3. A previous Docker image tag (locally)

Every release is built with a version tag, so an older version can be run at any time:

```bash
docker images crop-yield                    # list the versions you have
docker run -p 8000:8000 crop-yield:v1       # run an older version
```

To recreate an older version from git: `git checkout <commit-or-tag>` then
`docker build -t crop-yield:<tag> .` (any commit that contains the Dockerfile, i.e. from Milestone 2 on).

## When something goes wrong

| Symptom | Likely cause | What to do |
|---|---|---|
| Live URL takes about a minute, then works | Free service was asleep | Normal; wake it before demos |
| `/` shows "Not Found" | An old version (before the `/` redirect) is still serving | Wait for the deploy to finish, refresh with Ctrl+F5 |
| Build fails | See the end of the build log (Render → Deploys → the failed deploy) | Fix in a pull request; meanwhile roll back (section 1) |
| Build is very slow | Too many ML threads on the shared build machine | The Dockerfile sets `OMP_NUM_THREADS=1`; keep it |
| `/predict` returns 422 | Invalid input (e.g. unknown crop, value out of range) | The response says which field; fix the request |
| Container never becomes healthy | The app failed to start | Render → **Logs**; locally `docker logs <container>` |
