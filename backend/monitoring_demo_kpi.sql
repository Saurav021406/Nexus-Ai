-- Demo source KPI for testing Live Monitoring.
-- Replace only the text YOUR_AUTH_USER_UUID inside the quotes with the UUID
-- shown for your account under Supabase Dashboard > Authentication > Users.

create table if not exists public.monitoring_demo_kpi (
    user_id uuid primary key references auth.users(id) on delete cascade,
    monthly_revenue numeric not null
);

alter table public.monitoring_demo_kpi enable row level security;

drop policy if exists "Users can read their own demo KPI"
    on public.monitoring_demo_kpi;
create policy "Users can read their own demo KPI"
    on public.monitoring_demo_kpi
    for select to authenticated
    using (auth.uid() = user_id);

insert into public.monitoring_demo_kpi (user_id, monthly_revenue)
values ('YOUR_AUTH_USER_UUID', 800)
on conflict (user_id)
do update set monthly_revenue = excluded.monthly_revenue;

-- After creating a monitor with an "above 900" threshold, run this separately
-- in a new SQL query to trigger an in-app alert:
--
-- update public.monitoring_demo_kpi
-- set monthly_revenue = 950
-- where user_id = 'YOUR_AUTH_USER_UUID';
