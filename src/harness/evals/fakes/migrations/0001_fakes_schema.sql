-- Schema for eval-time fake adapters.
--
-- These tables live alongside the production `messages` / `*_summaries`
-- tables in the per-agent sqlite DB, but are only touched by the fake
-- adapter implementations in `harness.evals.fakes.*`. Applied by
-- `harness.evals.fakes.base.apply_migrations()` -- NOT by the production
-- `harness.core.storage.load()` path.

-- email --------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fake_email_thread (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    participants TEXT NOT NULL,     -- JSON array of email addresses
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fake_email_message (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES fake_email_thread(id),
    direction TEXT NOT NULL,        -- 'inbound' | 'outbound'
    from_addr TEXT NOT NULL,
    to_addrs TEXT NOT NULL,         -- JSON array
    cc_addrs TEXT NOT NULL DEFAULT '[]',
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    sent_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fake_email_msg_thread
    ON fake_email_message(thread_id, sent_at);

-- sms ----------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fake_sms_message (
    id TEXT PRIMARY KEY,
    contact_phone TEXT NOT NULL,
    direction TEXT NOT NULL,        -- 'inbound' | 'outbound'
    body TEXT NOT NULL,
    sent_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fake_sms_contact
    ON fake_sms_message(contact_phone, sent_at);

-- contacts -----------------------------------------------------------

CREATE TABLE IF NOT EXISTS fake_contact (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    phone TEXT,
    email TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- computer -----------------------------------------------------------

CREATE TABLE IF NOT EXISTS fake_computer_state (
    agent_id TEXT PRIMARY KEY,
    tmpdir TEXT NOT NULL,
    created_at TEXT NOT NULL
);
