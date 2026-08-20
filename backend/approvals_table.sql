-- Generic Human Approval table (Phase 4 roadmap item).
-- Run this once in your Supabase project's SQL editor.
--
-- Mirrors report_versions (which keeps working unchanged for reports),
-- generalized with a resource_type so any AI output can go through the
-- same approve/reject/version-history flow. Starting consumer:
-- resource_type = 'agent_workflow' (Multi-Agent results).

create table if not exists approvals (
    id uuid primary key default gen_random_uuid(),
    resource_type text not null,
    resource_id text not null,
    dataset_id text,
    user_id uuid not null,
    version_number integer not null,
    content jsonb not null,
    approval_status text not null default 'pending',
    rejection_reason text,
    created_at timestamptz not null default now()
);

create index if not exists approvals_resource_idx
    on approvals (resource_type, resource_id, user_id);

create index if not exists approvals_user_idx
    on approvals (user_id);

-- If you use Supabase Row Level Security, add a policy similar to
-- whatever you already have on report_versions, e.g.:
--
-- alter table approvals enable row level security;
--
-- create policy "Users can manage their own approvals"
--     on approvals
--     for all
--     using (auth.uid() = user_id)
--     with check (auth.uid() = user_id);
