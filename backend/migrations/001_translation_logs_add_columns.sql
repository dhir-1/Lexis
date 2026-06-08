-- Neon migration for audio transcript metadata.
-- Safe to run more than once.

ALTER TABLE translation_logs
    ADD COLUMN IF NOT EXISTS detected_language VARCHAR(10) DEFAULT NULL;

ALTER TABLE translation_logs
    ADD COLUMN IF NOT EXISTS raw_text TEXT DEFAULT NULL;
