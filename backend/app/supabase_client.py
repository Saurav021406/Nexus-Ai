from supabase import create_client, Client
from app.config import settings

# Use this for actions that should respect the calling user's row-level
# security (RLS) policies - i.e. anything scoped to "their own" data.
supabase_anon: Client = create_client(settings.supabase_url, settings.supabase_anon_key)

# Use this ONLY for trusted backend operations that must bypass RLS
# (e.g. system-level writes). Never expose this client or key to the frontend.
supabase_admin: Client = create_client(
    settings.supabase_url, settings.supabase_service_role_key
)
