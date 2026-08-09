# Nexus AI — Phase 1 (Foundation)

Goal of this phase: prove the full chain works — **React frontend → FastAPI backend → Supabase (auth + DB)** — before any AI is added. Nothing here is fake or a placeholder; once you plug in your own Supabase keys, sign-up/sign-in and the "Test full chain" button will genuinely work.

## 1. Create your Supabase project

1. Go to https://supabase.com, sign in, click **New project**.
2. Pick a name (e.g. `nexus-ai`), a strong DB password (save it somewhere), and a region close to you.
3. Once it's created, go to **Project Settings → API**. You'll need three values:
   - `Project URL`
   - `anon public` key
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
# now open .env and paste in your Supabase URL + anon key + service role key

uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs — you should see the FastAPI auto-generated API docs with `/health` and `/me` endpoints. `/health` should work immediately in the browser.

## 3. Frontend setup (VS Code, terminal 2 — separate terminal, backend keeps running)

```bash
cd frontend
npm install

cp .env.example .env
# paste in the SAME Supabase URL + anon key (not service role — anon only)
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
- **401 on /me**: session token might not be attached yet — refresh the page after signing in, or check that `VITE_SUPABASE_URL`/`VITE_SUPABASE_ANON_KEY` match your project exactly.
- **Backend won't start**: confirm `.env` exists in `backend/` (not just `.env.example`) and all three Supabase values are filled in.

## Next: Phase 2

Once this works reliably, the next phase adds the first real AI agent (RAG over PDFs + SQL querying over uploaded CSVs) using the OpenAI Agents SDK. Come back and we'll scaffold `backend/app/agents/` together.
