# One-Page Free Deployment (Render + Streamlit)

This guide gives you a simple free deployment for course demo use.

## What this setup does

- Backend API (FastAPI) on Render free tier.
- Frontend UI (Streamlit) on Streamlit Community Cloud free tier.
- Frontend calls backend via `API_URL`.

## Files already prepared in this repo

- `render.yaml` (Render Blueprint config)
- `requirements-deploy.txt` (lighter backend deploy deps)
- `frontend/requirements.txt` (frontend-only deps for Streamlit Cloud)

## Step 1: Push project to GitHub

1. Create a GitHub repository.
2. Push this project to that repository.

Example commands:

```bash
git add .
git commit -m "Add one-page free deployment bundle"
git push origin main
```

## Step 2: Deploy backend on Render

1. Open Render dashboard.
2. New -> Blueprint.
3. Connect your GitHub repository.
4. Render reads `render.yaml` and creates service automatically.
5. Wait until status is Live.

After success, test:

- `https://<your-render-domain>/health`
- `https://<your-render-domain>/version`

## Step 3: Deploy frontend on Streamlit Community Cloud

1. Open Streamlit Community Cloud.
2. New app.
3. Select the same repository.
4. Set Main file path to `frontend/app.py`.
5. In Advanced settings or Secrets/Env vars, set:

- `API_URL = https://<your-render-domain>`
- `API_TIMEOUT_SECONDS = 30`

6. Deploy.

## Step 4: Verify end-to-end

1. Open Streamlit app URL from any machine.
2. Submit one sample article.
3. Confirm UI returns prediction/explanation without backend error.

## Common free-tier issues

- Cold start: first request after idle may take longer.
- Build timeout: if Render build is slow, retry once.
- If Streamlit fails dependency install, keep app path as `frontend/app.py` and ensure `frontend/requirements.txt` is present.

## Optional hardening (not required for course demo)

- Add custom domain.
- Add rate limiting.
- Add persistent data store for logs/metrics.
