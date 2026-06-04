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

For Render, use the Supabase Direct connection URI, not the pooler.

Format:

```text
postgresql://postgres:<PASSWORD_URL_ENCODED>@db.<PROJECT_REF>.supabase.co:5432/postgres?sslmode=require
```

If your database password contains `@`, replace that character with `%40` in the URL.

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
DATABASE_URL=postgresql://postgres:<PASSWORD_URL_ENCODED>@db.<PROJECT_REF>.supabase.co:5432/postgres?sslmode=require
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
{"status":"ok","model_loaded":true}
```

If `model_loaded` is `false`, check Cloudinary model env vars and Render logs.

If Render build fails while building `dlib` with a CMake policy error, make sure your pushed `Dockerfile` starts with:

```dockerfile
FROM python:3.11-slim-bookworm
```

The Dockerfile also pins `cmake<4` and sets:

```dockerfile
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
```

The unpinned `python:3.11-slim` image can currently use a newer Debian/CMake combination that breaks `dlib==19.24.2`.

If you still see the same dlib error after this fix:

1. Confirm the Render build log begins with `FROM python:3.11-slim-bookworm`.
2. Confirm the log has a step installing `"cmake<4"`.
3. In Render, use **Manual Deploy -> Clear build cache & deploy**.
4. Make sure you pushed the latest `Dockerfile` to GitHub before redeploying.

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
- Render `/health` has `model_loaded: true`.
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
- Render free tier may cold-start slowly because TensorFlow, MediaPipe, dlib, and model loading are heavy.
- Browser webcam access requires HTTPS. Vercel provides HTTPS automatically.
- The `/api/register` endpoint triggers the original local registration script. That original script expects a local webcam and GUI, so it is preserved but not very useful on a headless Render server. Your deployed system should use the already-migrated Cloudinary faces/models for verification.

## Troubleshooting

Database connection fails:

- Make sure the password is URL-encoded.
- Use the Direct connection string.
- Add `?sslmode=require`.

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
