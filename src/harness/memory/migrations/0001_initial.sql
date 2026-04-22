-- Initial schema: message log + six tiered summary tables + migration audit.

CREATE TABLE IF NOT EXISTS applied_migrations (
    name TEXT PRIMARY KEY,
    applied_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    ts_ns INTEGER NOT NULL,
    role TEXT NOT NULL,
    content_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_ts ON messages (ts_ns);

CREATE TABLE IF NOT EXISTS one_minute_summaries (
    id TEXT PRIMARY KEY,
    date TEXT NOT NULL,        -- ISO date: 'YYYY-MM-DD'
    hour INTEGER NOT NULL,     -- 0-23
    minute INTEGER NOT NULL,   -- 0-59
    summary TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at_ns INTEGER NOT NULL,
    UNIQUE(date, hour, minute)
);
CREATE INDEX IF NOT EXISTS one_minute_summaries_dhm
    ON one_minute_summaries(date, hour, minute);

CREATE TABLE IF NOT EXISTS five_minute_summaries (
    id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    minute INTEGER NOT NULL,   -- 0, 5, 10, ..., 55
    summary TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at_ns INTEGER NOT NULL,
    UNIQUE(date, hour, minute)
);
CREATE INDEX IF NOT EXISTS five_minute_summaries_dhm
    ON five_minute_summaries(date, hour, minute);

CREATE TABLE IF NOT EXISTS hourly_summaries (
    id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    summary TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at_ns INTEGER NOT NULL,
    UNIQUE(date, hour)
);
CREATE INDEX IF NOT EXISTS hourly_summaries_dh
    ON hourly_summaries(date, hour);

CREATE TABLE IF NOT EXISTS daily_summaries (
    id TEXT PRIMARY KEY,
    date TEXT NOT NULL UNIQUE,
    summary TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS weekly_summaries (
    id TEXT PRIMARY KEY,
    week_start_date TEXT NOT NULL UNIQUE,
    summary TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS monthly_summaries (
    id TEXT PRIMARY KEY,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,    -- 1-12
    summary TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at_ns INTEGER NOT NULL,
    UNIQUE(year, month)
);
CREATE INDEX IF NOT EXISTS monthly_summaries_ym
    ON monthly_summaries(year, month);
