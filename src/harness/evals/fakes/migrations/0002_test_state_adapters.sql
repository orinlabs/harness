-- Additional eval-time fake adapter tables for Bedrock state adapters.
--
-- These are intentionally separate from 0001 so existing eval DBs that have
-- already applied fakes/0001 still pick up projects/documents on next startup.

-- documents ----------------------------------------------------------

CREATE TABLE IF NOT EXISTS fake_document (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'note',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fake_document_updated
    ON fake_document(updated_at);
CREATE INDEX IF NOT EXISTS idx_fake_document_kind_updated
    ON fake_document(kind, updated_at);

-- projects -----------------------------------------------------------

CREATE TABLE IF NOT EXISTS fake_project (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    visibility TEXT NOT NULL DEFAULT 'internal',
    status TEXT NOT NULL DEFAULT 'incomplete',
    parent_project_id TEXT REFERENCES fake_project(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fake_project_parent
    ON fake_project(parent_project_id, start_date);
CREATE INDEX IF NOT EXISTS idx_fake_project_status_visibility
    ON fake_project(status, visibility);
