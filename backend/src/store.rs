//! SQLite persistence: subtitle jobs, their cues, and the node's defaults.
//!
//! One `Arc<tokio::sync::Mutex<Connection>>` (the async mutex, matching `ryu-social`
//! / `ryu-teams` / `ryu-mail`) — a single writer with WAL underneath.
//!
//! # Why a job is a row and not a request
//!
//! Transcribing a feature film is minutes of local compute. A blocking
//! `POST /jobs` that returns when the file is ready would die on Core's proxy
//! timeout long before the work finished, and the user would have no way to learn
//! that the work continued. So a job is created, persisted, worked on in the
//! background and polled — and because it is persisted, a node that restarts
//! mid-film knows what was in flight rather than losing it silently. Jobs that were
//! running when the process died are marked interrupted at boot
//! ([`SubtitleStore::reset_interrupted`]) rather than left claiming to be at 40%
//! forever.
//!
//! # Cues live in the row
//!
//! The cue list is stored as JSON on the job, not as a `cues` table. It is written
//! once, read whole, and never queried across jobs — a table would buy indexes
//! nothing asks for, and a 90-minute film is ~1200 cues, well inside what SQLite
//! stores comfortably in one column. Keeping them together also makes the
//! re-render cheap: switching a finished job from SRT to WebVTT, or from
//! translated-only to bilingual, is a formatting pass over the stored cues with no
//! model call and no second transcription.
//!
//! # Sealed at rest
//!
//! The transcript (`cues`), the source video's path and file name, the path of
//! the copy written beside the video and the error text are sealed per value
//! (`encb:v1:`, bound to `jobs.<column>` and the job id) under this app's
//! plugin subkey (see [`crate::seal_sql`]). Status, progress, language, format,
//! counts and timestamps stay plaintext: the queue and the list order on them.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension, Row};
use ryu_crypto::content_sqlite::{ColumnPlan, KeyKind};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::cues::{Cue, Format, Layout};
use crate::seal_keys::PluginKeys;
use crate::seal_sql::{collect_readable, open_store, opened, opened_opt, SqlSeal};

/// The plugin id (the sealing key's scope).
pub const PLUGIN_ID: &str = "@ryu/subtitles";
/// The registered content-store name (part of every key derivation; never rename).
pub const STORE: &str = "subtitles.db";

/// The sealed `jobs` columns.
const SEALED_COLUMNS: [&str; 5] = ["source_path", "source_name", "error", "output_path", "cues"];

/// The sealed columns (`encb:v1:`, bound to `jobs.<column>` and the job id).
pub fn column_plans() -> Result<Vec<ColumnPlan>> {
    SEALED_COLUMNS
        .iter()
        .map(|column| ColumnPlan::text("jobs", column, "id", KeyKind::Text))
        .collect()
}

/// The job columns, with the sealed ones opened.
const JOB_COLUMNS: &str = "id, ryu_open('jobs.source_path', id, source_path),
                    ryu_open('jobs.source_name', id, source_name), target_language, format,
                    layout, status, progress, stage, ryu_open('jobs.error', id, error),
                    duration_ms, cue_count, translated_count, engine, model,
                    ryu_open('jobs.output_path', id, output_path), created_at, updated_at,
                    completed_at";

/// Schema version this build expects. Bump it and add an arm in
/// [`SubtitleStore::migrate`] when the shape changes.
const SCHEMA_VERSION: i32 = 1;

/// Where a job is in the pipeline. Persisted as the lowercase string, so a new
/// variant never renumbers an existing row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Accepted, waiting for the worker.
    Queued,
    /// Decoding + transcribing windows.
    Transcribing,
    /// Cues exist; the model is translating them.
    Translating,
    /// A subtitle file was written.
    Completed,
    /// It stopped, and `error` says why.
    Failed,
    /// The user cancelled it, or the process died mid-run.
    Canceled,
}

impl Status {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Transcribing => "transcribing",
            Self::Translating => "translating",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Canceled => "canceled",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "transcribing" => Self::Transcribing,
            "translating" => Self::Translating,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "canceled" => Self::Canceled,
            _ => Self::Queued,
        }
    }

    /// Whether the job is finished, in any sense. Used by the worker's cancellation
    /// check and by `retry`, which only makes sense on a terminal job.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Canceled)
    }
}

/// One subtitle job, as the API returns it.
///
/// `cues` is deliberately NOT part of this struct: the list view returns every job,
/// and shipping ~1200 cues per row would make it a megabyte-scale response. Cues are
/// fetched per job through [`SubtitleStore::cues`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    /// Absolute path of the source media on this machine.
    pub source_path: String,
    /// Basename, for display.
    pub source_name: String,
    /// BCP-47 tag of the language the cues are translated into.
    pub target_language: String,
    pub format: Format,
    pub layout: Layout,
    pub status: Status,
    /// 0.0–1.0 across the whole pipeline (transcription is the first 80%).
    pub progress: f64,
    /// Human-readable current step, e.g. "Transcribing 12:30 of 1:34:02".
    pub stage: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Media duration in ms, once the decoder knows it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    pub cue_count: u64,
    /// How many cues the model actually translated. Less than `cue_count` means a
    /// partly untranslated file, which the UI says out loud.
    pub translated_count: u64,
    /// STT engine used.
    pub engine: String,
    /// Translation model used.
    pub model: String,
    /// Where the subtitle file was written, when it was written beside the video.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
}

/// What a caller must supply to create a job. Everything optional is resolved
/// against [`Settings`] before it reaches here.
#[derive(Debug, Clone)]
pub struct NewJob {
    pub source_path: String,
    pub source_name: String,
    pub target_language: String,
    pub format: Format,
    pub layout: Layout,
    pub engine: String,
    pub model: String,
}

/// Node-wide defaults, edited in the app's settings tab and used to fill in a job
/// the caller under-specified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Default target language code.
    pub target_language: String,
    pub format: Format,
    pub layout: Layout,
    /// STT engine. `whisper` is the default and the only one that returns per-cue
    /// timings today (parakeet returns text only), so changing it is a documented
    /// downgrade rather than a preference.
    pub engine: String,
    /// Translation model; empty means the bundled local default.
    #[serde(default)]
    pub model: String,
    /// Whether a finished job also writes the subtitle file NEXT TO the video, which
    /// is what makes players (VLC, Plex, Jellyfin) pick it up with no further steps.
    pub write_beside_source: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            target_language: crate::languages::DEFAULT_TARGET.to_string(),
            format: Format::Srt,
            layout: Layout::Translated,
            engine: "whisper".to_string(),
            model: String::new(),
            write_beside_source: true,
        }
    }
}

/// SQLite-backed store. Cheap to clone (`Arc` inside).
#[derive(Clone)]
pub struct SubtitleStore {
    conn: Arc<Mutex<Connection>>,
    seal: SqlSeal,
}

impl SubtitleStore {
    /// Open (creating if needed) the database at `path`, resolving this
    /// process's sealing key with the database's directory as the plugin data
    /// dir (see [`crate::seal_keys`]).
    pub fn open(path: PathBuf) -> Result<Self> {
        let dir = path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let keys = PluginKeys::resolve(PLUGIN_ID, &dir)?;
        Self::open_with_keys(path, &keys)
    }

    /// [`Self::open`] with explicit keys.
    pub fn open_with_keys(path: PathBuf, keys: &PluginKeys) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn =
            Connection::open(&path).with_context(|| format!("opening {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL").ok();
        // WAL admits readers from OTHER processes (a `sqlite3` shell, a backup),
        // about which this process's mutex knows nothing.
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Self::from_connection(conn, keys, Some(&path))
    }

    /// In-memory store for tests, sealing under a random key.
    #[cfg(test)]
    pub fn memory() -> Result<Self> {
        use base64::Engine as _;
        use rand::RngCore as _;
        let mut root = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut root);
        let keys = PluginKeys::sealing(
            ryu_crypto::PluginCipher::from_subkey_b64(
                PLUGIN_ID,
                &base64::engine::general_purpose::STANDARD.encode(root),
            )?,
            crate::seal_keys::KeySource::Delivered,
        );
        Self::from_connection(Connection::open_in_memory()?, &keys, None)
    }

    fn from_connection(
        mut conn: Connection,
        keys: &PluginKeys,
        path: Option<&Path>,
    ) -> Result<Self> {
        // Before any statement that could rewrite a cell.
        ryu_crypto::content_sqlite::enable_secure_delete(&conn)?;
        Self::migrate_blocking(&conn)?;
        let seal = open_store(&mut conn, keys, STORE, &column_plans()?, path)
            .context("sealing subtitles.db")?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            seal,
        })
    }

    /// The sealing health item: `{state, detail?, unreadable_rows}`.
    pub fn seal_health(&self) -> serde_json::Value {
        self.seal.health()
    }

    /// Whether this store can write sealed values (or is in compat mode).
    fn writable(&self) -> bool {
        self.seal.current().ensure_writable().is_ok()
    }

    /// Synchronous migration, run once at open before anything can serve a request.
    fn migrate_blocking(conn: &Connection) -> Result<()> {
        let version: i32 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap_or(0);
        if version < 1 {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    format TEXT NOT NULL,
                    layout TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT '',
                    error TEXT,
                    duration_ms INTEGER,
                    cue_count INTEGER NOT NULL DEFAULT 0,
                    translated_count INTEGER NOT NULL DEFAULT 0,
                    engine TEXT NOT NULL DEFAULT 'whisper',
                    model TEXT NOT NULL DEFAULT '',
                    output_path TEXT,
                    cues TEXT NOT NULL DEFAULT '[]',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    completed_at INTEGER
                 );
                 CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs (created_at DESC);
                 CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    blob TEXT NOT NULL
                 );",
            )?;
        }
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(())
    }

    /// Insert a queued job and return it.
    pub async fn create_job(&self, new: NewJob) -> Result<Job> {
        let now = now_ms();
        let job = Job {
            id: format!("sub_{}", uuid::Uuid::new_v4().simple()),
            source_path: new.source_path,
            source_name: new.source_name,
            target_language: new.target_language,
            format: new.format,
            layout: new.layout,
            status: Status::Queued,
            progress: 0.0,
            stage: "Queued".to_string(),
            error: None,
            duration_ms: None,
            cue_count: 0,
            translated_count: 0,
            engine: new.engine,
            model: new.model,
            output_path: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
        };
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO jobs (id, source_path, source_name, target_language, format, layout,
                               status, progress, stage, engine, model, cues, created_at, updated_at)
             VALUES (?1, ryu_seal('jobs.source_path', ?1, ?2),
                     ryu_seal('jobs.source_name', ?1, ?3), ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
                     ryu_seal('jobs.cues', ?1, '[]'), ?12, ?13)",
            params![
                job.id,
                job.source_path,
                job.source_name,
                job.target_language,
                job.format.extension(),
                layout_str(job.layout),
                job.status.as_str(),
                job.progress,
                job.stage,
                job.engine,
                job.model,
                job.created_at,
                job.updated_at,
            ],
        )?;
        Ok(job)
    }

    /// Newest first. `limit` caps the list so a node with a year of jobs still
    /// answers the companion's first paint quickly.
    pub async fn list_jobs(&self, limit: usize) -> Result<Vec<Job>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(&format!(
            "SELECT {JOB_COLUMNS} FROM jobs ORDER BY created_at DESC LIMIT ?1"
        ))?;
        let rows = stmt.query_map(params![limit as i64], row_to_job)?;
        Ok(collect_readable(rows)?)
    }

    pub async fn get_job(&self, id: &str) -> Result<Option<Job>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(&format!("SELECT {JOB_COLUMNS} FROM jobs WHERE id = ?1"))?;
        Ok(stmt.query_row(params![id], row_to_job).optional()?)
    }

    /// The next queued job, oldest first — one at a time, deliberately: two
    /// concurrent transcriptions on a laptop make both slower and neither finishes
    /// sooner.
    ///
    /// A queued job whose sealed values do not open (counted) is skipped, and
    /// nothing is picked up while the store refuses writes (wrong or missing
    /// key): the worker could not record any progress.
    pub async fn next_queued(&self) -> Result<Option<Job>> {
        if !self.writable() {
            return Ok(None);
        }
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(&format!(
            "SELECT {JOB_COLUMNS} FROM jobs WHERE status = 'queued' ORDER BY created_at ASC"
        ))?;
        let rows = stmt.query_map([], row_to_job)?;
        for row in rows {
            match row {
                Ok(job) => return Ok(Some(job)),
                Err(error) if crate::seal_sql::is_unreadable(&error) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(None)
    }

    /// Move a job's status, optionally recording an error.
    pub async fn set_status(&self, id: &str, status: Status, error: Option<&str>) -> Result<()> {
        let conn = self.conn.lock().await;
        let completed = status.is_terminal().then(now_ms);
        // `COALESCE(completed_at, ?5)` — the EXISTING stamp wins. The reverse order
        // moves the stamp every time a terminal job is re-stated (a retry that fails
        // again, a cancel after a failure), which would make "finished at" mean
        // "last touched at" and quietly break any ordering built on it.
        conn.execute(
            "UPDATE jobs SET status = ?2, error = ryu_seal('jobs.error', ?1, ?3), updated_at = ?4,
                             completed_at = COALESCE(completed_at, ?5)
             WHERE id = ?1",
            params![id, status.as_str(), error, now_ms(), completed],
        )?;
        Ok(())
    }

    /// Move a worker-owned job only while it is still active. A cancellation or
    /// deletion that wins the race is terminal and must never be overwritten by
    /// a late transcription/translation result.
    pub async fn set_status_if_active(
        &self,
        id: &str,
        status: Status,
        error: Option<&str>,
    ) -> Result<bool> {
        let conn = self.conn.lock().await;
        let completed = status.is_terminal().then(now_ms);
        Ok(conn.execute(
            "UPDATE jobs SET status = ?2, error = ryu_seal('jobs.error', ?1, ?3), updated_at = ?4,
                             completed_at = COALESCE(completed_at, ?5)
             WHERE id = ?1 AND status NOT IN ('completed', 'failed', 'canceled')",
            params![id, status.as_str(), error, now_ms(), completed],
        )? > 0)
    }

    /// Atomically request cancellation only if the worker has not already
    /// committed a terminal outcome.
    pub async fn cancel_if_active(&self, id: &str, reason: &str) -> Result<bool> {
        let conn = self.conn.lock().await;
        Ok(conn.execute(
            "UPDATE jobs SET status = 'canceled', error = ryu_seal('jobs.error', ?1, ?2),
                             updated_at = ?3,
                             completed_at = COALESCE(completed_at, ?3)
             WHERE id = ?1 AND status NOT IN ('completed', 'failed', 'canceled')",
            params![id, reason, now_ms()],
        )? > 0)
    }

    /// Commit completion as one compare-and-set. The caller may publish output
    /// only when this succeeds.
    pub async fn complete_if_active(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().await;
        Ok(conn.execute(
            "UPDATE jobs SET status = 'completed', progress = 1, stage = 'Done',
                             error = NULL, updated_at = ?2,
                             completed_at = COALESCE(completed_at, ?2)
             WHERE id = ?1 AND status NOT IN ('completed', 'failed', 'canceled')",
            params![id, now_ms()],
        )? > 0)
    }

    /// Record progress. Cheap and called often, so it writes only the four columns
    /// that change.
    pub async fn set_progress(
        &self,
        id: &str,
        progress: f64,
        stage: &str,
        duration_ms: Option<u64>,
    ) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE jobs SET progress = ?2, stage = ?3,
                             duration_ms = COALESCE(?4, duration_ms), updated_at = ?5
             WHERE id = ?1 AND status NOT IN ('completed', 'failed', 'canceled')",
            params![
                id,
                progress.clamp(0.0, 1.0),
                stage,
                duration_ms.map(|d| d as i64),
                now_ms()
            ],
        )?;
        Ok(())
    }

    /// Store the finished cue list and the counts derived from it.
    pub async fn set_cues(&self, id: &str, cues: &[Cue]) -> Result<()> {
        let json = serde_json::to_string(cues)?;
        let translated = cues.iter().filter(|c| c.translated.is_some()).count() as i64;
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE jobs SET cues = ryu_seal('jobs.cues', ?1, ?2), cue_count = ?3, translated_count = ?4, updated_at = ?5
             WHERE id = ?1 AND status NOT IN ('completed', 'failed', 'canceled')",
            params![id, json, cues.len() as i64, translated, now_ms()],
        )?;
        Ok(())
    }

    pub async fn cues(&self, id: &str) -> Result<Vec<Cue>> {
        let conn = self.conn.lock().await;
        let json: Option<String> = conn
            .query_row(
                "SELECT ryu_open('jobs.cues', id, cues) FROM jobs WHERE id = ?1",
                params![id],
                |r| opened(r, 0),
            )
            .optional()?;
        let Some(json) = json else {
            return Ok(Vec::new());
        };
        Ok(serde_json::from_str(&json).unwrap_or_default())
    }

    /// Record where the file was written beside the source video.
    pub async fn set_output_path(&self, id: &str, path: Option<&str>) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE jobs SET output_path = ryu_seal('jobs.output_path', ?1, ?2), updated_at = ?3
             WHERE id = ?1",
            params![id, path, now_ms()],
        )?;
        Ok(())
    }

    /// Change the rendering choices on a finished job, so re-rendering to WebVTT or
    /// to bilingual costs no model call.
    pub async fn set_render_options(&self, id: &str, format: Format, layout: Layout) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE jobs SET format = ?2, layout = ?3, updated_at = ?4 WHERE id = ?1",
            params![id, format.extension(), layout_str(layout), now_ms()],
        )?;
        Ok(())
    }

    /// Put a terminal job back in the queue, clearing its previous outcome.
    pub async fn requeue(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE jobs SET status = 'queued', progress = 0, stage = 'Queued', error = NULL,
                             cues = ryu_seal('jobs.cues', ?1, '[]'), cue_count = 0, translated_count = 0,
                             completed_at = NULL, updated_at = ?2
             WHERE id = ?1",
            params![id, now_ms()],
        )?;
        Ok(())
    }

    pub async fn delete_job(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().await;
        Ok(conn.execute("DELETE FROM jobs WHERE id = ?1", params![id])? > 0)
    }

    /// At boot, anything that claims to be running is a lie: the process that was
    /// running it is gone. Mark those canceled so the list does not show a job stuck
    /// at 40% forever, and so `retry` is offered on them.
    ///
    /// On a store that refuses writes (wrong or missing key) the error text is
    /// cleared instead of sealed: the stage already says what happened, and
    /// startup must not fail on it.
    pub async fn reset_interrupted(&self) -> Result<usize> {
        let error = self
            .writable()
            .then_some("The node restarted while this job was running.");
        let conn = self.conn.lock().await;
        Ok(conn.execute(
            "UPDATE jobs SET status = 'canceled', stage = 'Interrupted by a restart',
                             error = CASE WHEN ?2 IS NULL THEN NULL
                                          ELSE ryu_seal('jobs.error', id, ?2) END,
                             updated_at = ?1
             WHERE status IN ('transcribing', 'translating')",
            params![now_ms(), error],
        )?)
    }

    pub async fn settings(&self) -> Result<Settings> {
        let conn = self.conn.lock().await;
        let blob: Option<String> = conn
            .query_row("SELECT blob FROM settings WHERE id = 1", [], |r| r.get(0))
            .optional()?;
        Ok(blob
            .and_then(|b| serde_json::from_str(&b).ok())
            .unwrap_or_default())
    }

    pub async fn save_settings(&self, settings: &Settings) -> Result<()> {
        let blob = serde_json::to_string(settings)?;
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO settings (id, blob) VALUES (1, ?1)
             ON CONFLICT(id) DO UPDATE SET blob = excluded.blob",
            params![blob],
        )?;
        Ok(())
    }

    /// Cheap readiness probe for `/health`: proves the DB is READABLE, not merely
    /// that the process is alive.
    pub async fn job_count(&self) -> Result<u64> {
        let conn = self.conn.lock().await;
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM jobs", [], |r| r.get(0))?;
        Ok(count as u64)
    }
}

fn layout_str(layout: Layout) -> &'static str {
    match layout {
        Layout::Translated => "translated",
        Layout::Source => "source",
        Layout::Bilingual => "bilingual",
    }
}

fn parse_layout(value: &str) -> Layout {
    match value {
        "source" => Layout::Source,
        "bilingual" => Layout::Bilingual,
        _ => Layout::Translated,
    }
}

fn row_to_job(row: &Row<'_>) -> rusqlite::Result<Job> {
    let format: String = row.get(4)?;
    let layout: String = row.get(5)?;
    let status: String = row.get(6)?;
    let duration: Option<i64> = row.get(10)?;
    Ok(Job {
        id: row.get(0)?,
        source_path: opened(row, 1)?,
        source_name: opened(row, 2)?,
        target_language: row.get(3)?,
        format: Format::parse(&format).unwrap_or_default(),
        layout: parse_layout(&layout),
        status: Status::parse(&status),
        progress: row.get(7)?,
        stage: row.get(8)?,
        error: opened_opt(row, 9)?,
        duration_ms: duration.map(|d| d.max(0) as u64),
        cue_count: row.get::<_, i64>(11)?.max(0) as u64,
        translated_count: row.get::<_, i64>(12)?.max(0) as u64,
        engine: row.get(13)?,
        model: row.get(14)?,
        output_path: opened_opt(row, 15)?,
        created_at: row.get(16)?,
        updated_at: row.get(17)?,
        completed_at: row.get(18)?,
    })
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_job(name: &str) -> NewJob {
        NewJob {
            source_path: format!("/movies/{name}.mp4"),
            source_name: format!("{name}.mp4"),
            target_language: "es".to_string(),
            format: Format::Srt,
            layout: Layout::Translated,
            engine: "whisper".to_string(),
            model: String::new(),
        }
    }

    fn cue(start_ms: u64, text: &str, translated: Option<&str>) -> Cue {
        Cue {
            start_ms,
            end_ms: start_ms + 1_000,
            text: text.to_string(),
            translated: translated.map(std::string::ToString::to_string),
        }
    }

    #[tokio::test]
    async fn a_new_job_starts_queued_at_zero() {
        let store = SubtitleStore::memory().expect("store");
        let job = store.create_job(new_job("film")).await.expect("create");
        assert_eq!(job.status, Status::Queued);
        assert_eq!(job.progress, 0.0);
        assert!(job.id.starts_with("sub_"));

        let loaded = store.get_job(&job.id).await.expect("get").expect("present");
        assert_eq!(loaded.source_name, "film.mp4");
        assert_eq!(loaded.target_language, "es");
    }

    #[tokio::test]
    async fn jobs_list_newest_first_and_respect_the_limit() {
        let store = SubtitleStore::memory().expect("store");
        for i in 0..3 {
            store
                .create_job(new_job(&format!("f{i}")))
                .await
                .expect("create");
            // Timestamps are millisecond-granular; without a nudge the ORDER BY is
            // a coin flip and this test would be flaky rather than wrong.
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        let all = store.list_jobs(10).await.expect("list");
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].source_name, "f2.mp4");
        assert_eq!(store.list_jobs(1).await.expect("list").len(), 1);
    }

    #[tokio::test]
    async fn progress_is_clamped_and_duration_is_sticky() {
        let store = SubtitleStore::memory().expect("store");
        let job = store.create_job(new_job("film")).await.expect("create");
        store
            .set_progress(&job.id, 5.0, "Transcribing", Some(120_000))
            .await
            .expect("progress");
        let loaded = store.get_job(&job.id).await.expect("get").expect("present");
        assert_eq!(loaded.progress, 1.0, "progress must never exceed 1.0");
        assert_eq!(loaded.duration_ms, Some(120_000));

        // A later update without a duration must not erase the one we know.
        store
            .set_progress(&job.id, 0.5, "Translating", None)
            .await
            .expect("progress");
        let loaded = store.get_job(&job.id).await.expect("get").expect("present");
        assert_eq!(loaded.duration_ms, Some(120_000));
    }

    #[tokio::test]
    async fn cues_round_trip_and_drive_the_counts() {
        let store = SubtitleStore::memory().expect("store");
        let job = store.create_job(new_job("film")).await.expect("create");
        let cues = vec![cue(0, "hello", Some("hola")), cue(2_000, "goodbye", None)];
        store.set_cues(&job.id, &cues).await.expect("set cues");

        let loaded = store.get_job(&job.id).await.expect("get").expect("present");
        assert_eq!(loaded.cue_count, 2);
        assert_eq!(loaded.translated_count, 1, "only one cue was translated");
        assert_eq!(store.cues(&job.id).await.expect("cues"), cues);
    }

    #[tokio::test]
    async fn terminal_status_stamps_completed_at_once() {
        let store = SubtitleStore::memory().expect("store");
        let job = store.create_job(new_job("film")).await.expect("create");
        store
            .set_status(&job.id, Status::Completed, None)
            .await
            .expect("status");
        let first = store.get_job(&job.id).await.expect("get").expect("present");
        assert!(first.completed_at.is_some());

        store
            .set_status(&job.id, Status::Failed, Some("boom"))
            .await
            .expect("status");
        let second = store.get_job(&job.id).await.expect("get").expect("present");
        assert_eq!(
            second.completed_at, first.completed_at,
            "stamp must not move"
        );
        assert_eq!(second.error.as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn canceled_or_deleted_jobs_reject_every_late_worker_transition() {
        let store = SubtitleStore::memory().expect("store");
        let canceled = store.create_job(new_job("canceled")).await.expect("create");
        assert!(store
            .cancel_if_active(&canceled.id, "Cancelled.")
            .await
            .expect("cancel"));

        assert!(!store
            .set_status_if_active(&canceled.id, Status::Translating, None)
            .await
            .expect("transition"));
        store
            .set_progress(&canceled.id, 1.0, "Done", None)
            .await
            .expect("progress");
        store
            .set_cues(&canceled.id, &[cue(0, "late", Some("tarde"))])
            .await
            .expect("cues");
        assert!(!store
            .complete_if_active(&canceled.id)
            .await
            .expect("complete"));

        let after = store
            .get_job(&canceled.id)
            .await
            .expect("get")
            .expect("present");
        assert_eq!(after.status, Status::Canceled);
        assert_eq!(after.stage, "Queued");
        assert_eq!(after.progress, 0.0);
        assert_eq!(after.error.as_deref(), Some("Cancelled."));
        assert!(store.cues(&canceled.id).await.expect("cues").is_empty());

        let deleted = store.create_job(new_job("deleted")).await.expect("create");
        assert!(store.delete_job(&deleted.id).await.expect("delete"));
        assert!(!store
            .complete_if_active(&deleted.id)
            .await
            .expect("complete deleted"));
    }

    #[tokio::test]
    async fn requeue_clears_the_previous_outcome() {
        let store = SubtitleStore::memory().expect("store");
        let job = store.create_job(new_job("film")).await.expect("create");
        store
            .set_cues(&job.id, &[cue(0, "hi", Some("hola"))])
            .await
            .expect("cues");
        store
            .set_status(&job.id, Status::Failed, Some("boom"))
            .await
            .expect("status");

        store.requeue(&job.id).await.expect("requeue");
        let loaded = store.get_job(&job.id).await.expect("get").expect("present");
        assert_eq!(loaded.status, Status::Queued);
        assert_eq!(loaded.cue_count, 0);
        assert!(loaded.error.is_none());
        assert!(loaded.completed_at.is_none());
        assert!(store.cues(&job.id).await.expect("cues").is_empty());
    }

    #[tokio::test]
    async fn next_queued_is_oldest_first_and_skips_running_jobs() {
        let store = SubtitleStore::memory().expect("store");
        let first = store.create_job(new_job("first")).await.expect("create");
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let second = store.create_job(new_job("second")).await.expect("create");

        assert_eq!(
            store.next_queued().await.expect("next").map(|j| j.id),
            Some(first.id.clone())
        );
        store
            .set_status(&first.id, Status::Transcribing, None)
            .await
            .expect("status");
        assert_eq!(
            store.next_queued().await.expect("next").map(|j| j.id),
            Some(second.id)
        );
    }

    #[tokio::test]
    async fn a_restart_cancels_jobs_that_claimed_to_be_running() {
        let store = SubtitleStore::memory().expect("store");
        let running = store.create_job(new_job("running")).await.expect("create");
        let queued = store.create_job(new_job("queued")).await.expect("create");
        store
            .set_status(&running.id, Status::Transcribing, None)
            .await
            .expect("status");

        assert_eq!(store.reset_interrupted().await.expect("reset"), 1);
        let after = store
            .get_job(&running.id)
            .await
            .expect("get")
            .expect("present");
        assert_eq!(after.status, Status::Canceled);
        assert!(after.error.is_some());
        // A queued job was never running, so it stays queued and still gets worked.
        let untouched = store
            .get_job(&queued.id)
            .await
            .expect("get")
            .expect("present");
        assert_eq!(untouched.status, Status::Queued);
    }

    #[tokio::test]
    async fn settings_default_until_saved_then_round_trip() {
        let store = SubtitleStore::memory().expect("store");
        let defaults = store.settings().await.expect("settings");
        assert_eq!(defaults.engine, "whisper");
        assert!(defaults.write_beside_source);

        let saved = Settings {
            target_language: "ja".into(),
            format: Format::Vtt,
            layout: Layout::Bilingual,
            engine: "whisper".into(),
            model: "qwen3-4b".into(),
            write_beside_source: false,
        };
        store.save_settings(&saved).await.expect("save");
        let loaded = store.settings().await.expect("settings");
        assert_eq!(loaded.target_language, "ja");
        assert_eq!(loaded.format, Format::Vtt);
        assert_eq!(loaded.layout, Layout::Bilingual);
        assert!(!loaded.write_beside_source);
    }

    #[tokio::test]
    async fn re_rendering_options_do_not_touch_the_cues() {
        let store = SubtitleStore::memory().expect("store");
        let job = store.create_job(new_job("film")).await.expect("create");
        let cues = vec![cue(0, "hello", Some("hola"))];
        store.set_cues(&job.id, &cues).await.expect("cues");

        store
            .set_render_options(&job.id, Format::Vtt, Layout::Bilingual)
            .await
            .expect("render options");
        let loaded = store.get_job(&job.id).await.expect("get").expect("present");
        assert_eq!(loaded.format, Format::Vtt);
        assert_eq!(loaded.layout, Layout::Bilingual);
        assert_eq!(store.cues(&job.id).await.expect("cues"), cues);
    }

    #[tokio::test]
    async fn deleting_is_idempotent_from_the_callers_point_of_view() {
        let store = SubtitleStore::memory().expect("store");
        let job = store.create_job(new_job("film")).await.expect("create");
        assert!(store.delete_job(&job.id).await.expect("delete"));
        assert!(!store.delete_job(&job.id).await.expect("delete"));
        assert!(store.get_job(&job.id).await.expect("get").is_none());
    }

    mod at_rest {
        use super::*;
        use crate::seal_keys::KeySource;
        use base64::Engine as _;
        use ryu_crypto::{ContentCipher, PluginCipher};

        const LINE: &str = "quokka transcript marker";
        const FILM: &str = "Zebra Holiday";

        fn keys(n: u8) -> PluginKeys {
            PluginKeys::sealing(
                PluginCipher::from_subkey_b64(
                    PLUGIN_ID,
                    &base64::engine::general_purpose::STANDARD.encode([n; 32]),
                )
                .unwrap(),
                KeySource::Delivered,
            )
        }

        fn db() -> PathBuf {
            std::env::temp_dir()
                .join(format!("ryu-subtitles-at-rest-{}", uuid::Uuid::new_v4()))
                .join(STORE)
        }

        fn on_disk(path: &Path, needle: &str) -> bool {
            let mut bytes = std::fs::read(path).unwrap_or_default();
            bytes.extend(std::fs::read(format!("{}-wal", path.display())).unwrap_or_default());
            bytes.windows(needle.len()).any(|w| w == needle.as_bytes())
        }

        async fn seed(store: &SubtitleStore) -> (String, String) {
            let first = store.create_job(new_job(FILM)).await.unwrap();
            store
                .set_cues(&first.id, &[cue(0, LINE, Some("hola"))])
                .await
                .unwrap();
            let second = store.create_job(new_job("other")).await.unwrap();
            (first.id, second.id)
        }

        fn raw(path: &Path) -> Vec<String> {
            let conn = Connection::open(path).unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT source_path || '|' || source_name || '|' || cues FROM jobs ORDER BY id",
                )
                .unwrap();
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .map(Result::unwrap)
                .collect();
            rows
        }

        #[tokio::test]
        async fn transcripts_and_paths_are_sealed_on_disk_and_read_back() {
            let path = db();
            let store = SubtitleStore::open_with_keys(path.clone(), &keys(1)).unwrap();
            let (id, _) = seed(&store).await;
            let job = store.get_job(&id).await.unwrap().unwrap();
            assert_eq!(job.source_name, format!("{FILM}.mp4"));
            assert_eq!(store.cues(&id).await.unwrap()[0].text, LINE);
            store
                .set_status(&id, Status::Failed, Some("quokka failure"))
                .await
                .unwrap();
            drop(store);
            for needle in [LINE, FILM, "quokka failure"] {
                assert!(!on_disk(&path, needle), "{needle} on disk");
            }
            let conn = Connection::open(&path).unwrap();
            let cipher = keys(1).content_cipher(STORE).unwrap();
            ryu_crypto::content_sqlite::assert_all_open(&conn, &cipher, &column_plans().unwrap())
                .unwrap();
        }

        #[tokio::test]
        async fn legacy_jobs_are_sealed_once_and_still_read() {
            let path = db();
            let (id, _) =
                seed(&SubtitleStore::open_with_keys(path.clone(), &PluginKeys::compat()).unwrap())
                    .await;
            assert!(on_disk(&path, LINE));
            let store = SubtitleStore::open_with_keys(path.clone(), &keys(2)).unwrap();
            assert_eq!(store.cues(&id).await.unwrap()[0].text, LINE);
            drop(store);
            assert!(!on_disk(&path, LINE));
            assert!(!on_disk(&path, FILM));
            let first = raw(&path);
            let conn = Connection::open(&path).unwrap();
            let cues: Vec<String> = conn
                .prepare("SELECT cues FROM jobs")
                .unwrap()
                .query_map([], |row| row.get(0))
                .unwrap()
                .map(Result::unwrap)
                .collect();
            assert!(cues
                .iter()
                .all(|value| ContentCipher::is_sealed_text(value)));
            drop(conn);
            drop(SubtitleStore::open_with_keys(path.clone(), &keys(2)).unwrap());
            assert_eq!(raw(&path), first);
        }

        #[tokio::test]
        async fn a_copied_value_is_skipped_and_other_keys_refuse_writes() {
            let path = db();
            let store = SubtitleStore::open_with_keys(path.clone(), &keys(3)).unwrap();
            let (first, second) = seed(&store).await;
            store
                .conn
                .lock()
                .await
                .execute(
                    "UPDATE jobs SET source_name = (SELECT source_name FROM jobs WHERE id = ?1)
                     WHERE id = ?2",
                    params![first, second],
                )
                .unwrap();
            let jobs = store.list_jobs(10).await.unwrap();
            assert_eq!(jobs.len(), 1);
            assert_eq!(jobs[0].id, first);
            assert_eq!(store.seal_health()["unreadable_rows"], 1);
            drop(store);
            for other in [keys(4), PluginKeys::compat()] {
                let wrong = SubtitleStore::open_with_keys(path.clone(), &other).unwrap();
                assert_eq!(wrong.seal_health()["state"], "unhealthy");
                assert!(wrong.create_job(new_job("refused")).await.is_err());
                // Nothing is picked up, and the startup reset never fails.
                assert!(wrong.next_queued().await.unwrap().is_none());
                wrong.reset_interrupted().await.unwrap();
            }
        }

        #[tokio::test]
        async fn an_interrupted_job_resets_on_an_unhealthy_store() {
            let path = db();
            let store = SubtitleStore::open_with_keys(path.clone(), &keys(5)).unwrap();
            let (id, _) = seed(&store).await;
            store
                .set_status(&id, Status::Transcribing, None)
                .await
                .unwrap();
            drop(store);
            let wrong = SubtitleStore::open_with_keys(path.clone(), &keys(6)).unwrap();
            assert_eq!(wrong.reset_interrupted().await.unwrap(), 1);
            drop(wrong);
            let store = SubtitleStore::open_with_keys(path, &keys(5)).unwrap();
            let job = store.get_job(&id).await.unwrap().unwrap();
            assert_eq!(job.status, Status::Canceled);
            assert_eq!(job.error, None);
        }
    }
}
