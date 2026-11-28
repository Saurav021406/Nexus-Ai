# Nexus AI

This project has grown beyond the original foundation phase and includes AutoML, RAG, and additional AI tools. The setup steps below describe the local development environment.

## 1. Create your Supabase project

1. Go to https://supabase.com, sign in, click **New project**.
2. Pick a name (e.g. `nexus-ai`), a strong DB password (save it somewhere), and a region close to you.
3. Once it's created, go to **Project Settings → API**. You'll need:
   - `Project URL`
   - `publishable` key (preferred; the app retains support for the legacy `anon public` key)
   - `service_role` key (keep this secret — backend only, never in frontend code)
4. Go to **Authentication → Providers** and make sure **Email** is enabled (it is by default). For now leave "Confirm email" on or off, your choice — off is easier while developing.

## 2. Backend setup (VS Code, terminal 1)

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# set SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, and SUPABASE_SERVICE_ROLE_KEY

uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs — you should see the FastAPI auto-generated API docs with `/health` and `/me` endpoints. `/health` should work immediately in the browser.

## 3. Frontend setup (VS Code, terminal 2 — separate terminal, backend keeps running)

```bash
cd frontend
npm install

cp .env.example .env
# set the SAME Supabase URL + publishable key (never the service-role key)
# VITE_API_BASE_URL should stay http://localhost:8000 for local dev

npm run dev
```

Visit http://localhost:5173. You should see a login screen.

## 4. Verify the full chain

1. On the login screen, click "Don't have an account? Sign up", enter any email/password (6+ chars), sign up.
2. If email confirmation is off, you'll be logged in immediately. If it's on, check the email Supabase sends (via the Supabase dashboard's Auth logs if using a fake email).
3. Once logged in, click **"Test full chain"**. You should see: `Backend confirmed you are: your@email.com (id: ...)`.

If you see that message, Phase 1 is done: auth, frontend, backend, and database are all correctly wired together.

## Troubleshooting

- **CORS error in browser console**: check `CORS_ORIGINS` in `backend/.env` includes `http://localhost:5173`.
- **401 on /me**: session token might not be attached yet — refresh the page after signing in, or check that `VITE_SUPABASE_URL`/`VITE_SUPABASE_PUBLISHABLE_KEY` match your project exactly.
- **Backend won't start**: confirm `.env` exists in `backend/` (not just `.env.example`) and all three Supabase values are filled in.

## Live Monitoring email alerts

The Live Monitoring tab creates in-app alerts when a KPI crosses its configured threshold. Browser notifications are optional and require granting permission in the tab; they appear while Nexus AI is open.

To enable the per-monitor **Also email me** option, set these values in `backend/.env` and restart the backend:

```env
SMTP_HOST=
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_FROM_EMAIL=
```

The SMTP server must support STARTTLS on the configured port. Email alerts are sent to the signed-in Supabase account address and only for monitors where email delivery was explicitly enabled. SMTP credentials belong only in the backend environment; do not put them in the frontend or commit them.

Run `backend/monitoring_tables.sql` in the Supabase SQL Editor to create or update the monitoring tables before creating monitors. `backend/monitoring_demo_kpi.sql` provides a sample KPI table for testing..
