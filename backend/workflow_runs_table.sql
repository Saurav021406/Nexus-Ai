-- Workflow persistence table (Phase 4 roadmap: "Persist workflow data",
-- Section 28). Run this once in your Supabase project's SQL editor.
--
-- One row per /agent/run or /agent/run/stream call. The full WorkflowState
-- snapshot (state.to_dict()) goes in the `state` jsonb column - traces,
-- tasks, specialist results, review, security, everything - so a past run
-- can be fully reconstructed and displayed later, not just its final
-- answer. Indexed columns are pulled out of that snapshot for fast
-- listing/filtering without needing to query into the jsonb blob.

create table if not exists workflow_runs (
    workflow_id uuid primary key,
    user_id uuid not null,
    dataset_id text,
    user_query text,
    goal text,
    status text not null,
    state jsonb not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists workflow_runs_user_idx
    on workflow_runs (user_id, created_at desc);

create index if not exists workflow_runs_dataset_idx
    on workflow_runs (dataset_id);

-- If you use Supabase Row Level Security (same pattern as approvals_table.sql):
--
-- alter table workflow_runs enable row level security;
--
-- create policy "Users can manage their own workflow runs"
--     on workflow_runs
--     for all
--     using (auth.uid() = user_id)
--     with check (auth.uid() = user_id);
