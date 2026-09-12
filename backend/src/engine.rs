//! The job worker: one queued job at a time, decode → transcribe → translate →
//! write.
//!
//! # One at a time, on purpose
//!
//! Two transcriptions running at once on a laptop do not finish sooner — they share
//! the same CPU (and, with a local whisper.cpp, the same model instance), so both run
//! at half speed and the user waits longer for the first file. The queue is
//! therefore serial, and a second job says "Queued" rather than lying about
//! progress.
//!
//! # Backpressure is the memory bound
//!
//! Decoding is CPU-bound and synchronous, so it runs in `spawn_blocking` and hands
//! windows over a bounded channel. The bound is what keeps a 2-hour film out of
//! memory: the decoder is faster than local transcription, so without it the decoder
//! would race to the end of the file and buffer the whole thing. With it, the
//! decoder blocks after [`WINDOW_QUEUE`] windows and memory stays at a handful of
//! megabytes regardless of the film's length.
//!
//! # Failure posture
//!
//! Transcription failures are FATAL to the job (a subtitle file missing its middle
//! ten minutes is not a subtitle file). Translation failures are NOT: an
//! untranslated cue keeps its source text, the job completes, and
//! `translated_count < cue_count` is what the UI reports. Writing beside the source
//! is also non-fatal — a read-only external drive still gets a downloadable file
//! from the app.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ryu_stt::SttHost;
use tokio::sync::{mpsc, Notify};

use crate::cues::{self, Cue};
use crate::library;
use crate::media::{self, Window};
use crate::store::{Job, Status, SubtitleStore};
use crate::translate::{self, GatewayConfig};

/// How many decoded windows may wait for the engine. Two is enough to keep the STT
/// engine fed while the decoder works ahead, and small enough to bound memory at
/// ~2 MB of PCM.
const WINDOW_QUEUE: usize = 2;

/// Fraction of the progress bar transcription owns. Translation is the rest, minus
/// the sliver for writing the file.
const TRANSCRIBE_SHARE: f64 = 0.8;

/// The local whisper.cpp voice server default (mirrors Core's `WHISPER_ADDR`).
const DEFAULT_WHISPER_URL: &str = "http://127.0.0.1:8090";

/// This app's plugin id — the `x-ryu-plugin-id` the app-event emitter presents.
pub const PLUGIN_ID: &str = "@ryu/subtitles";

/// Everything the worker needs, cloned into the background task.
#[derive(Clone)]
pub struct Worker {
    pub store: SubtitleStore,
    pub http: reqwest::Client,
    pub events: Arc<ryu_app_events::EventEmitter>,
    /// Woken when a job is created, so a new job starts now rather than on the next
    /// poll tick.
    pub wake: Arc<Notify>,
}

/// The sidecar's [`SttHost`]: the whisper.cpp base URL and the Gateway
/// url/bearer, resolved from the environment Core injects at spawn. Same code path
/// Core runs in-process — this is a wiring shim, not a second STT implementation.
struct SidecarSttHost;

impl SttHost for SidecarSttHost {
    fn whisper_base_url(&self) -> String {
        std::env::var("RYU_WHISPER_URL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_WHISPER_URL.to_string())
    }

    fn audio_cpp_base_url(&self) -> String {
        std::env::var("RYU_AUDIOCPP_URL")
            .ok()
            .map(|value| value.trim().trim_end_matches('/').to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| {
                let port = std::env::var("RYU_AUDIOCPP_PORT")
                    .ok()
                    .and_then(|value| value.trim().parse::<u16>().ok())
                    .unwrap_or_else(|| {
                        let offset = std::env::var("RYU_PORT_OFFSET")
                            .ok()
                            .and_then(|value| value.trim().parse::<u16>().ok())
                            .unwrap_or(0);
                        8086u16.saturating_add(offset)
                    });
                format!("http://127.0.0.1:{port}")
            })
    }

    fn gateway_url(&self) -> String {
        std::env::var("RYU_GATEWAY_URL")
            .ok()
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| translate::DEFAULT_GATEWAY_URL.to_string())
    }

    fn gateway_bearer(&self) -> Result<String, String> {
        // The local gateway accepts the `ryu-local` dev bearer; a configured token
        // wins. The sidecar is a local data plane, so there is no remote-fleet
        // fail-closed branch here.
        Ok(std::env::var("RYU_GATEWAY_TOKEN")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "ryu-local".to_string()))
    }

    fn parakeet_model_dir(&self) -> PathBuf {
        crate::paths::ryu_dir().join("models").join("parakeet-v3")
    }
}

/// Run the queue for the process lifetime. Returns a handle the shutdown path
/// aborts, so a supervised restart never runs two workers over one database.
pub fn spawn(worker: Worker) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // Anything that claimed to be running belonged to a process that is gone.
        if let Ok(n) = worker.store.reset_interrupted().await {
            if n > 0 {
                tracing::warn!("subtitles: {n} job(s) were interrupted by a restart");
            }
        }
        loop {
            match worker.store.next_queued().await {
                Ok(Some(job)) => {
                    let id = job.id.clone();
                    if let Err(e) = run_job(&worker, job).await {
                        tracing::warn!(job = %id, error = %e, "subtitles: job failed");
                        let message = e.to_string();
                        let transitioned = worker
                            .store
                            .set_status_if_active(&id, Status::Failed, Some(&message))
                            .await
                            .unwrap_or(false);
                        if transitioned {
                            worker
                                .events
                                .emit_with_notify(
                                    "@ryu/subtitles#job.failed",
                                    serde_json::json!({ "job_id": id, "error": message }),
                                    Some(
                                        ryu_app_events::NotifyHint::info(
                                            "Subtitle job failed",
                                            Some(message.clone()),
                                        )
                                        .with_level("error"),
                                    ),
                                )
                                .await;
                        }
                    }
                }
                Ok(None) => {
                    // Idle: wait to be woken by a new job, but re-poll on a timer
                    // anyway so a notification lost to a race cannot strand the
                    // queue.
                    tokio::select! {
                        () = worker.wake.notified() => {}
                        () = tokio::time::sleep(std::time::Duration::from_secs(5)) => {}
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "subtitles: could not read the job queue");
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                }
            }
        }
    })
}

/// Run one job end to end. Errors here are the job's `error` field.
async fn run_job(worker: &Worker, job: Job) -> anyhow::Result<()> {
    let store = &worker.store;
    let id = job.id.clone();
    if !store
        .set_status_if_active(&id, Status::Transcribing, None)
        .await?
    {
        return Ok(());
    }
    store
        .set_progress(&id, 0.0, "Reading the video", None)
        .await?;

    let source = PathBuf::from(&job.source_path);
    // Re-validate at run time, not just at creation: a queued job may have waited
    // while the user moved or deleted the file.
    library::validate_source(&job.source_path).map_err(|e| anyhow::anyhow!("{e}"))?;

    let language = crate::languages::find(&job.target_language)
        .ok_or_else(|| anyhow::anyhow!("unknown target language `{}`", job.target_language))?;

    // ---- transcription -------------------------------------------------------
    let (tx, mut rx) = mpsc::channel::<Result<Window, String>>(WINDOW_QUEUE);
    let decode_path = source.clone();
    let progress_store = store.clone();
    let progress_id = id.clone();
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<(u64, Option<u64>)>();

    let decoder = tokio::task::spawn_blocking(move || {
        let sender = tx;
        let result = media::decode_windows(
            &decode_path,
            |window| sender.blocking_send(Ok(window)).is_ok(),
            |done_ms, total_ms| {
                let _ = progress_tx.send((done_ms, total_ms));
            },
        );
        if let Err(e) = result {
            let _ = sender.blocking_send(Err(e.to_string()));
        }
    });

    // Progress is reported off the decode thread so a slow SQLite write never
    // stalls decoding.
    let progress_task = tokio::spawn(async move {
        let mut last_write = std::time::Instant::now();
        while let Some((done_ms, total_ms)) = progress_rx.recv().await {
            // One write per second at most: a 2-hour film emits ~240 windows, and
            // every one of them writing twice is pure churn.
            if last_write.elapsed() < std::time::Duration::from_millis(900) {
                continue;
            }
            last_write = std::time::Instant::now();
            let fraction = total_ms
                .filter(|t| *t > 0)
                .map_or(0.0, |t| (done_ms as f64 / t as f64).clamp(0.0, 1.0));
            let stage = match total_ms {
                Some(total) => format!("Transcribing {} of {}", clock(done_ms), clock(total)),
                None => format!("Transcribing {}", clock(done_ms)),
            };
            let _ = progress_store
                .set_progress(&progress_id, fraction * TRANSCRIBE_SHARE, &stage, total_ms)
                .await;
        }
    });

    let host = SidecarSttHost;
    let mut all_cues: Vec<Cue> = Vec::new();
    let mut window_index = 0usize;
    while let Some(message) = rx.recv().await {
        let window = message.map_err(|e| anyhow::anyhow!("{e}"))?;
        // Cancellation is checked between windows: the granularity is one window
        // (~30 s of audio, a second or two of compute), which is responsive without
        // polling the database inside the engine call.
        if is_canceled(store, &id).await {
            drop(rx);
            progress_task.abort();
            let _ = decoder.await;
            return Ok(());
        }

        let transcription = ryu_stt::transcribe_wav_detailed(
            &worker.http,
            &host,
            window.wav,
            format!("window-{window_index}.wav"),
            Some(&job.engine),
        )
        .await
        .map_err(|e| anyhow::anyhow!("transcription failed: {e}"))?;

        all_cues.extend(cues::cues_from_segments(
            window.offset_ms,
            &transcription.segments,
            &transcription.text,
            window.duration_ms,
        ));
        window_index += 1;
    }
    progress_task.abort();
    decoder.await.ok();

    if is_canceled(store, &id).await {
        return Ok(());
    }

    let mut all_cues = cues::normalize(all_cues);
    if all_cues.is_empty() {
        anyhow::bail!("no speech was found in this file");
    }
    store.set_cues(&id, &all_cues).await?;

    // ---- translation ---------------------------------------------------------
    // Skipped entirely when the transcript is already in the target language: a
    // model asked to translate English into English rewrites it, which shows up as
    // subtitles that do not match the audio.
    let needs_translation = !transcript_is_target(&all_cues, language);
    if needs_translation {
        if !store
            .set_status_if_active(&id, Status::Translating, None)
            .await?
        {
            return Ok(());
        }
        let config = GatewayConfig::from_env(Some(job.model.clone()).filter(|m| !m.is_empty()));
        let progress_store = store.clone();
        let progress_id = id.clone();
        let translated = translate::translate_cues(
            &worker.http,
            &config,
            language,
            &mut all_cues,
            || is_canceled(store, &id),
            move |done, total| {
                let store = progress_store.clone();
                let id = progress_id.clone();
                let fraction = if total == 0 {
                    1.0
                } else {
                    done as f64 / total as f64
                };
                let stage = format!("Translating line {done} of {total}");
                tokio::spawn(async move {
                    let _ = store
                        .set_progress(
                            &id,
                            TRANSCRIBE_SHARE + fraction * (0.98 - TRANSCRIBE_SHARE),
                            &stage,
                            None,
                        )
                        .await;
                });
            },
        )
        .await;
        let Some(translated) = translated else {
            return Ok(());
        };
        if is_canceled(store, &id).await {
            return Ok(());
        }
        store.set_cues(&id, &all_cues).await?;
        if translated == 0 {
            // Every batch AND every single-line retry failed — the gateway is down
            // or the model is missing. The transcript is real work and is kept, but
            // calling this "completed" would be a lie about a file that is entirely
            // in the wrong language.
            anyhow::bail!(
                "the transcript was created, but no line could be translated — check that the local gateway is running"
            );
        }
    }

    // ---- write ---------------------------------------------------------------
    store
        .set_progress(&id, 0.98, "Writing the subtitle file", None)
        .await?;
    if is_canceled(store, &id).await {
        return Ok(());
    }
    let rendered = cues::render(&all_cues, job.format, job.layout);

    // The data-dir copy is what the app downloads. Always written, so a job on a
    // read-only volume is still usable.
    let data_path = data_output_path(&id, job.format.extension());
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&data_path, rendered.as_bytes())?;

    // Completion is the commit point. If cancellation or deletion won after the
    // last check, discard the just-written in-app artifact and publish nothing.
    if !store.complete_if_active(&id).await? {
        remove_data_output(&id);
        return Ok(());
    }

    // The copy BESIDE the video is what makes a player pick it up automatically.
    // Best-effort: an external drive mounted read-only must not fail the job.
    let settings = store.settings().await.unwrap_or_default();
    let mut beside: Option<String> = None;
    if settings.write_beside_source {
        let target =
            library::sidecar_output_path(&source, &job.target_language, job.format.extension());
        match std::fs::write(&target, rendered.as_bytes()) {
            Ok(()) => beside = Some(target.to_string_lossy().into_owned()),
            Err(e) => tracing::warn!(
                path = %target.display(),
                error = %e,
                "subtitles: could not write beside the source; the download still works"
            ),
        }
    }
    store.set_output_path(&id, beside.as_deref()).await?;

    worker
        .events
        .emit_with_notify(
            "@ryu/subtitles#job.completed",
            serde_json::json!({
                "job_id": id,
                "source_path": job.source_path,
                "source_name": job.source_name,
                "target_language": job.target_language,
                "format": job.format.extension(),
                "cue_count": all_cues.len(),
                "output_path": beside,
            }),
            // A long transcription finishing is exactly the job-done moment worth a
            // push: the companion is not necessarily open when it lands.
            Some(ryu_app_events::NotifyHint::info(
                "Subtitles ready",
                Some(format!(
                    "{} — {} cues in {}",
                    job.source_name,
                    all_cues.len(),
                    job.format.extension()
                )),
            )),
        )
        .await;
    Ok(())
}

/// Whether the user cancelled (or deleted) the job while it was running.
async fn is_canceled(store: &SubtitleStore, id: &str) -> bool {
    match store.get_job(id).await {
        Ok(Some(job)) => job.status == Status::Canceled,
        // A job that vanished was deleted mid-run; stopping is the right answer.
        Ok(None) => true,
        // A database read failure cannot prove the job is still active. Stop
        // rather than spending more compute or publishing a late artifact.
        Err(_) => true,
    }
}

/// Heuristic for "the audio is already in the target language", so the translation
/// pass can be skipped.
///
/// Deliberately crude, because the cost of each mistake is asymmetric: skipping a
/// needed translation leaves an obviously-untranslated file the user can retry,
/// while running an unneeded one silently paraphrases correct subtitles. So this
/// only fires for the case it can actually be sure about — the engine returned
/// Latin-script text and the target is English, which is where the false-positive
/// rate of a script check is lowest and the "transcribe an English film for English
/// subtitles" workflow is most common.
fn transcript_is_target(cues: &[Cue], language: &crate::languages::Language) -> bool {
    if language.code != "en" {
        return false;
    }
    let sample: String = cues
        .iter()
        .take(20)
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    if sample.trim().is_empty() {
        return false;
    }
    let letters = sample.chars().filter(|c| c.is_alphabetic()).count();
    if letters == 0 {
        return false;
    }
    let ascii = sample
        .chars()
        .filter(|c| c.is_alphabetic() && c.is_ascii())
        .count();
    ascii * 100 / letters >= 95
}

/// Where the downloadable copy lives: `<data dir>/subtitles/<job id>.<ext>`.
pub fn data_output_path(job_id: &str, extension: &str) -> PathBuf {
    crate::paths::ryu_dir()
        .join("subtitles")
        .join(format!("{job_id}.{extension}"))
}

/// `H:MM:SS` (or `M:SS` under an hour) for the progress line.
fn clock(ms: u64) -> String {
    let total = ms / 1000;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

/// Delete the generated file for a job, best-effort. Called when the job row goes.
pub fn remove_data_output(job_id: &str) {
    for extension in ["srt", "vtt"] {
        let path = data_output_path(job_id, extension);
        if path.exists() {
            std::fs::remove_file(path).ok();
        }
    }
}

/// Whether the source file for `path` still exists — the list view greys out jobs
/// whose video has since been moved.
#[must_use]
pub fn source_exists(path: &str) -> bool {
    Path::new(path).is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(text: &str) -> Cue {
        Cue {
            start_ms: 0,
            end_ms: 1_000,
            text: text.to_string(),
            translated: None,
        }
    }

    #[test]
    fn clock_switches_to_hours_only_when_there_are_hours() {
        assert_eq!(clock(0), "0:00");
        assert_eq!(clock(65_000), "1:05");
        assert_eq!(clock(3_725_000), "1:02:05");
    }

    #[test]
    fn english_latin_transcripts_skip_the_translation_pass() {
        let english = crate::languages::find("en").expect("en");
        let cues = vec![cue("the quick brown fox"), cue("jumps over the lazy dog")];
        assert!(transcript_is_target(&cues, english));
    }

    #[test]
    fn a_non_latin_transcript_is_never_assumed_to_be_english() {
        let english = crate::languages::find("en").expect("en");
        let cues = vec![cue("こんにちは、元気ですか"), cue("これはテストです")];
        assert!(!transcript_is_target(&cues, english));
    }

    #[test]
    fn a_non_english_target_always_translates() {
        let spanish = crate::languages::find("es").expect("es");
        let cues = vec![cue("the quick brown fox")];
        assert!(
            !transcript_is_target(&cues, spanish),
            "an English transcript still needs translating into Spanish"
        );
    }

    #[test]
    fn an_empty_transcript_does_not_short_circuit_translation() {
        let english = crate::languages::find("en").expect("en");
        assert!(!transcript_is_target(&[], english));
        assert!(!transcript_is_target(&[cue("   ")], english));
    }

    #[test]
    fn data_output_paths_are_per_job_and_per_format() {
        let srt = data_output_path("sub_abc", "srt");
        let vtt = data_output_path("sub_abc", "vtt");
        assert!(srt.ends_with("subtitles/sub_abc.srt"));
        assert!(vtt.ends_with("subtitles/sub_abc.vtt"));
    }

    #[tokio::test]
    async fn a_deleted_job_reads_as_canceled_so_the_worker_stops() {
        let store = SubtitleStore::memory().expect("store");
        assert!(is_canceled(&store, "sub_missing").await);
    }
}
