-- Live KPI monitoring persistence. Run this once in the Supabase SQL Editor.

create table if not exists monitoring_rules (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    name text not null,
    source_table text not null,
    metric_column text not null,
    comparison text not null check (comparison in ('above', 'below')),
    threshold double precision not null,
    last_value double precision,
    last_checked_at timestamptz,
    email_enabled boolean not null default false,
    is_triggered boolean not null default false,
    is_active boolean not null default true,
    created_at timestamptz not null default now()
);

create index if not exists monitoring_rules_active_idx
    on monitoring_rules (is_active, last_checked_at);
create index if not exists monitoring_rules_user_idx
    on monitoring_rules (user_id, created_at desc);

create table if not exists monitoring_alerts (
    id uuid primary key default gen_random_uuid(),
    rule_id uuid not null references monitoring_rules(id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    rule_name text not null,
    metric_value double precision not null,
    threshold double precision not null,
    comparison text not null check (comparison in ('above', 'below')),
    message text not null,
    email_status text not null default 'not_requested'
        check (email_status in ('not_requested', 'pending', 'sent', 'failed')),
    created_at timestamptz not null default now(),
    acknowledged_at timestamptz
);

alter table monitoring_rules
    add column if not exists email_enabled boolean not null default false;
alter table monitoring_alerts
    add column if not exists email_status text not null default 'not_requested';

create index if not exists monitoring_alerts_user_idx
    on monitoring_alerts (user_id, created_at desc);

alter table monitoring_rules enable row level security;
alter table monitoring_alerts enable row level security;

drop policy if exists "Users manage their own monitoring rules" on monitoring_rules;
create policy "Users manage their own monitoring rules"
    on monitoring_rules for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists "Users manage their own monitoring alerts" on monitoring_alerts;
create policy "Users manage their own monitoring alerts"
    on monitoring_alerts for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);
