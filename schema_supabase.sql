-- Supabase/PostgreSQL schema for Customer Segmentation app
-- Run in Supabase SQL editor.

begin;

create extension if not exists pgcrypto;

-- Users for app login
create table if not exists public.users (
  id bigserial primary key,
  username varchar(100) not null unique,
  email varchar(255) not null unique,
  password text not null,
  created_at timestamptz not null default now()
);

-- Advertisement templates
create table if not exists public.ads (
  id bigserial primary key,
  cluster integer not null,
  ad_text text not null,
  image_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Customer predictions and captured features
create table if not exists public.customers (
  id bigserial primary key,
  email varchar(255) not null,
  balance numeric(14, 2) not null,
  purchases numeric(14, 2) not null,
  cash_advance numeric(14, 2) not null,
  credit_limit numeric(14, 2) not null,
  payments numeric(14, 2) not null,
  full_payment numeric(5, 4) not null,
  purchases_freq numeric(5, 4) not null,
  cash_adv_freq numeric(5, 4) not null,
  cluster integer not null,
  created_at timestamptz not null default now(),
  constraint customers_email_chk check (position('@' in email) > 1),
  constraint customers_freq_chk check (
    full_payment between 0 and 1
    and purchases_freq between 0 and 1
    and cash_adv_freq between 0 and 1
  )
);

-- Email/campaign dispatch logs
create table if not exists public.logs (
  id bigserial primary key,
  customer_id bigint references public.customers(id) on delete set null,
  ad_id bigint references public.ads(id) on delete set null,
  email varchar(255) not null,
  "timestamp" timestamptz not null default now()
);

-- Helpful indexes
create index if not exists idx_ads_cluster on public.ads(cluster);
create index if not exists idx_customers_cluster on public.customers(cluster);
create index if not exists idx_customers_email on public.customers(email);
create index if not exists idx_logs_timestamp on public.logs("timestamp" desc);
create index if not exists idx_logs_customer_id on public.logs(customer_id);
create index if not exists idx_logs_ad_id on public.logs(ad_id);

-- Keep updated_at current
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_ads_updated_at on public.ads;
create trigger trg_ads_updated_at
before update on public.ads
for each row
execute function public.set_updated_at();

-- Optional: RLS disabled for this app-level auth design
alter table public.users disable row level security;
alter table public.ads disable row level security;
alter table public.customers disable row level security;
alter table public.logs disable row level security;

commit;
