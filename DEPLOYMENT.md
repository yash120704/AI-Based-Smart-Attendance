# Smart Attendance Cloud Deployment Guide

This guide deploys the project as:

- Frontend dashboard: Vercel / Next.js
- Backend API: Render / FastAPI / Docker
- Database: Supabase PostgreSQL
- File/model storage: Cloudinary

## Current Status

The code is deployment-ready from the repo side.

Already completed:

- Supabase tables were created and verified: `attendance`, `persons`.
- Cloudinary migration uploaded 30 face images.
- Cloudinary migration uploaded 3 model files:
  - `global_behavior.h5`
  - `global_label_map.pkl`
  - `global_preprocessor.pkl`
- Cloudinary read-back was verified:
  - `Krissh_Verma`: 10 face images
  - `Raghav_Sejpal`: 10 face images
  - `Yash_Kashyap`: 10 face images
- Frontend checks passed:
  - `npm run lint`
  - `npm run build`
- Python syntax checks passed for the modified backend/cloud files.

Important: real secrets are not stored in the repo. They must be added in Render, Vercel, or a local `.env` file that you do not commit.

## Where Env Vars Go

There are three different env places:

1. Render backend env vars
   These are set in the Render dashboard for the FastAPI service.

2. Vercel frontend env vars
   These are set in the Vercel dashboard for the Next.js app.

3. Local env files
   These are only for running locally. Use `.env.example` as a template for backend cloud-mode testing and `frontend/.env.local.example` for frontend local testing.

Do not paste real secrets into `.env.example`; it is only a template.

## Step 1: Supabase

This was already done once with your provided database URL.

If you ever need to run it again, open Supabase SQL Editor and run:

```sql
CREATE TABLE IF NOT EXISTS persons (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    registered_at TIMESTAMP DEFAULT NOW(),
    total_attendances INTEGER DEFAULT 0,
    blocked BOOLEAN DEFAULT FALSE,
    blocked_until TIMESTAMP
);

CREATE TABLE IF NOT EXISTS attendance (
    id SERIAL PRIMARY KEY,
    person_name TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    face_confidence REAL,
    behavior_confidence REAL,
    is_proxy BOOLEAN DEFAULT FALSE,
    alert_message TEXT,
    attempts INTEGER DEFAULT 1,
    status TEXT,
    blink_detected BOOLEAN DEFAULT FALSE
);
```

For Render, use the Supabase **Session Pooler** URI unless your Supabase project has the paid IPv4 add-on.

Format:

```text
postgresql://postgres.<PROJECT_REF>:<PASSWORD_URL_ENCODED>@aws-0-<REGION>.pooler.supabase.com:5432/postgres?sslmode=require
```

If your database password contains `@`, replace that character with `%40` in the URL.

Why Session Pooler: Supabase Direct connections use `db.<project-ref>.supabase.co:5432`, which resolves to IPv6 on many projects. Render services may not have outbound IPv6, so the app can fail with `Network is unreachable`. Supabase documents Session Pooler as the persistent-backend option for IPv4-only networks.

## Step 2: Cloudinary

This was already done once.

Uploaded:

- 30 face images under `smart-attendance/faces/<person_name>/`
- 3 model files under `smart-attendance/models/`

If you ever need to rerun migration locally:

```powershell
$env:CLOUDINARY_CLOUD_NAME="your-cloud-name"
$env:CLOUDINARY_API_KEY="your-api-key"
$env:CLOUDINARY_API_SECRET="your-api-secret"
python scripts/migrate_to_cloud.py
```

## Step 3: Push Code To GitHub

From the repo root:

```powershell
git status
git add .
git commit -m "Deploy smart attendance cloud stack"
git push
```

## Step 4: Deploy Backend On Render

Use Render Docker deployment because this project needs native dependencies for face recognition, MediaPipe, OpenCV, and TensorFlow.

Render setup:

1. Go to Render.
2. New Web Service.
3. Connect your GitHub repo.
4. Root directory: repo root.
5. Runtime / Language: Docker.
6. Render should detect the root `Dockerfile`.
7. Health check path: `/health`.

Set these Render env vars:

```text
DATABASE_URL=postgresql://postgres.<PROJECT_REF>:<PASSWORD_URL_ENCODED>@aws-0-<REGION>.pooler.supabase.com:5432/postgres?sslmode=require
CLOUDINARY_CLOUD_NAME=<your-cloud-name>
CLOUDINARY_API_KEY=<your-api-key>
CLOUDINARY_API_SECRET=<your-api-secret>
ALLOWED_ORIGIN=https://your-vercel-app.vercel.app
SECRET_KEY=<generate-any-random-string>
```

Deploy the service.

After deploy, test:

```text
https://your-render-service.onrender.com/health
```

Expected:

```json
{"status":"ok","model_loaded":false,"vision_loaded":false}
```

The backend lazy-loads the heavy face recognition, MediaPipe, and TensorFlow stack on the first `/api/verify-frame` request. This keeps Render from running out of memory before the web server opens its port. After the first successful verify request, `vision_loaded` and `model_loaded` should become `true`.

If Render build fails while building `dlib` with a CMake policy error, make sure your pushed `Dockerfile` starts with:

```dockerfile
FROM python:3.11-slim-bookworm
```

The Dockerfile also pins `cmake<4` and sets:

```dockerfile
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
```

The unpinned `python:3.11-slim` image can currently use a newer Debian/CMake combination that breaks `dlib==19.24.2`.

The API requirements use `dlib-bin==19.24.2` plus `face_recognition==1.3.0 --no-deps` inside the Dockerfile. The Dockerfile also installs `face_recognition_models` directly from `https://github.com/ageitgey/face_recognition_models` and verifies the import after `COPY . .`. This avoids compiling `dlib` from source on Render while still making the `dlib` module and model files available to `face_recognition`.

The Dockerfile pins `setuptools==70.3.0` because `face_recognition_models` imports `pkg_resources`, which is no longer available in newer setuptools releases.

If you still see the same dlib error after this fix:

1. Confirm the Render build log begins with `FROM python:3.11-slim-bookworm`.
2. Confirm the log installs `dlib-bin==19.24.2`.
3. Confirm the log installs `git+https://github.com/ageitgey/face_recognition_models`.
4. In Render, use **Manual Deploy -> Clear build cache & deploy**.
5. Make sure you pushed the latest `Dockerfile` to GitHub before redeploying.

## Step 5: Deploy Frontend On Vercel

Vercel setup:

1. Go to Vercel.
2. Add New Project.
3. Import the same GitHub repo.
4. Set root directory to `frontend`.
5. Framework preset: Next.js.

Set this Vercel env var:

```text
NEXT_PUBLIC_API_URL=https://your-render-service.onrender.com
```

Set it in Vercel Project Settings -> Environment Variables. Do not create a Vercel secret named `api_url`; `frontend/vercel.json` does not reference that secret.

Deploy.

## Step 6: Update Render CORS

After Vercel deploys, copy the final Vercel URL.

Go back to Render and set:

```text
ALLOWED_ORIGIN=https://your-real-vercel-url.vercel.app
```

Redeploy or restart the Render service.

This matters because the backend intentionally accepts browser API calls only from the configured frontend origin.

## Step 7: Final Verification

Check these in order:

- Render `/health` returns `status: ok`.
- Render `/health` opens quickly. It may show `vision_loaded: false` before webcam verification because the ML stack loads lazily.
- Vercel dashboard opens.
- Dashboard stats load without CORS errors.
- Attendance history page loads.
- Persons page loads.
- Block / unblock / re-enable buttons work.
- Alerts page loads.
- Verify page asks for camera permission.
- Verify page sends frames to `/api/verify-frame`.
- `/api/verify-frame` returns a stage such as `FACE`, `BLINK`, `BEHAVIOR`, `SUCCESS`, `RETRY`, or `BLOCKED`.

## Important Notes

- Local mode still works without `DATABASE_URL`; it uses SQLite and local files.
- Cloud mode activates when `DATABASE_URL` and Cloudinary env vars are set.
- Render free tier has only 512 MiB RAM. The backend now lazy-loads TensorFlow, MediaPipe, and face recognition on `/api/verify-frame`; dashboard/database endpoints should run, but webcam verification may still need a larger Render instance if memory runs out during the first verification.
- Browser webcam access requires HTTPS. Vercel provides HTTPS automatically.
- The `/api/register` endpoint triggers the original local registration script. That original script expects a local webcam and GUI, so it is preserved but not very useful on a headless Render server. Your deployed system should use the already-migrated Cloudinary faces/models for verification.

## Troubleshooting

Database connection fails:

- Make sure the password is URL-encoded.
- On Render, use the Supabase Session Pooler connection string, not Direct connection.
- Add `?sslmode=require`.
- If you see an IPv6 address and `Network is unreachable`, you are still using the Direct connection string.

CORS error in browser:

- Make sure Render `ALLOWED_ORIGIN` exactly matches the Vercel URL.
- Include `https://`.
- Do not include a trailing slash.

Model not loaded:

- Confirm the three model files exist in Cloudinary under `smart-attendance/models/`.
- Confirm Render has all `CLOUDINARY_*` env vars.
- Check Render logs during startup.

Camera not opening:

- Use the deployed HTTPS Vercel URL.
- Allow camera permission in the browser.
- Check `frontend/vercel.json` has the camera permissions policy.

Frontend cannot call backend:

- Confirm `NEXT_PUBLIC_API_URL` in Vercel points to the Render service URL.
- Redeploy Vercel after changing env vars.
