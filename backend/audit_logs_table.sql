-- Authenticated users can read their own audit history.
-- The backend service role writes events; users cannot edit or delete them.

create table if not exists audit_logs (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    action text not null,
    endpoint text not null,
    status_code smallint not null,
    duration_ms integer not null check (duration_ms >= 0),
    created_at timestamptz not null default now()
);

create index if not exists audit_logs_user_created_idx
    on audit_logs (user_id, created_at desc);

alter table audit_logs enable row level security;

drop policy if exists "Users can read their own audit logs" on audit_logs;
create policy "Users can read their own audit logs"
    on audit_logs
    for select to authenticated
    using (auth.uid() = user_id);
