-- Multi-address contacts for harness fake adapters.
--
-- Replaces scalar `fake_contact.phone` / `fake_contact.email` with
-- normalized child tables that carry per-address verification status,
-- mirroring the production `defaults_contact_phone` / `_email` schema in
-- bedrock-api. The fake DB is per-eval and ephemeral, so we hard-cut over
-- (no compatibility mirror).

CREATE TABLE IF NOT EXISTS fake_contact_phone (
    id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL REFERENCES fake_contact(id) ON DELETE CASCADE,
    phone TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    is_primary INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'verified',          -- 'verified' | 'pending'
    verification_source TEXT NOT NULL DEFAULT '',
    verified_at TEXT,
    last_seen_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Index used by tool list/search; no global uniqueness, since the harness
-- does not have a per-agent uniqueness constraint to enforce.
CREATE INDEX IF NOT EXISTS idx_fake_contact_phone_contact
    ON fake_contact_phone(contact_id, is_primary, created_at);

CREATE TABLE IF NOT EXISTS fake_contact_email (
    id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL REFERENCES fake_contact(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    is_primary INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'verified',
    verification_source TEXT NOT NULL DEFAULT '',
    verified_at TEXT,
    last_seen_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fake_contact_email_contact
    ON fake_contact_email(contact_id, is_primary, created_at);

-- Backfill from the existing scalar columns so old eval DBs keep working
-- through the cutover, then drop the scalars.
INSERT INTO fake_contact_phone (
    id, contact_id, phone, is_primary, status, verification_source,
    verified_at, created_at, updated_at
)
SELECT
    'sim_phone_' || substr(hex(randomblob(8)), 1, 12),
    id,
    phone,
    1,
    'verified',
    'legacy_backfill',
    created_at,
    created_at,
    updated_at
FROM fake_contact
WHERE phone IS NOT NULL AND phone != '';

INSERT INTO fake_contact_email (
    id, contact_id, email, is_primary, status, verification_source,
    verified_at, created_at, updated_at
)
SELECT
    'sim_email_' || substr(hex(randomblob(8)), 1, 12),
    id,
    email,
    1,
    'verified',
    'legacy_backfill',
    created_at,
    created_at,
    updated_at
FROM fake_contact
WHERE email IS NOT NULL AND email != '';

ALTER TABLE fake_contact DROP COLUMN phone;
ALTER TABLE fake_contact DROP COLUMN email;
