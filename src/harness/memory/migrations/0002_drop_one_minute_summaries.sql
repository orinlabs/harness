-- The 1-minute summary tier is removed. The 5-minute tier now reads
-- directly from the raw `messages` log (same pattern the v2 hourly
-- path was already using) so the 1m rollup is no longer needed --
-- and running it was firing an LLM call per completed minute, which
-- meant agents were burning ~60 summary calls/hour even when nothing
-- interesting had happened.
--
-- Drop the table and its index. Safe to re-apply: `IF EXISTS`
-- tolerates running against a DB that already has them missing
-- (e.g. a brand-new agent created after this migration, whose
-- initial schema path wouldn't have created them in the first place
-- once 0001 is eventually edited; or a DB that had them dropped
-- manually for testing).

DROP INDEX IF EXISTS one_minute_summaries_dhm;
DROP TABLE IF EXISTS one_minute_summaries;
