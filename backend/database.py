from __future__ import annotations

import os
import threading
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import errorcodes

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def get_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured.")
    return psycopg2.connect(database_url)


def log_to_database(
    input_type: str,
    text: str,
    confidence: float | None = None,
    detected_language: str | None = None,
    raw_text: str | None = None,
):
    conn = None
    cursor = None

    # Try the new schema first, then fall back to the old three-column layout.
    insert_with_new_columns = """
        INSERT INTO translation_logs (
            input_type,
            translated_text,
            confidence_score,
            detected_language,
            raw_text
        )
        VALUES (%s, %s, %s, %s, %s)
    """
    insert_legacy = """
        INSERT INTO translation_logs (
            input_type,
            translated_text,
            confidence_score
        )
        VALUES (%s, %s, %s)
    """

    try:
        conn = get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                insert_with_new_columns,
                (input_type, text, confidence, detected_language, raw_text),
            )
        except psycopg2.Error as exc:
            conn.rollback()
            if exc.pgcode != errorcodes.UNDEFINED_COLUMN:
                raise

            # Backward compatibility: keep working until the Neon migration lands.
            cursor.execute(insert_legacy, (input_type, text, confidence))

        conn.commit()
    except Exception as exc:
        print(f"[DB ERROR]: {exc}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def log_async(
    input_type: str,
    text: str,
    confidence: float | None = None,
    detected_language: str | None = None,
    raw_text: str | None = None,
):
    thread = threading.Thread(
        target=log_to_database,
        kwargs={
            "input_type": input_type,
            "text": text,
            "confidence": confidence,
            "detected_language": detected_language,
            "raw_text": raw_text,
        },
        daemon=True,
    )
    thread.start()
