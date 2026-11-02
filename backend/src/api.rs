//! The HTTP surface, mounted at `/api/subtitles`.
//!
//! Paths here are RELATIVE to that mount, and every one of them must also appear in
//! `sidecars[0].http.routes[]` in the manifest — that list is an ALLOWLIST matched by
//! exact segment count, so a route declared as `/jobs/:id` does not admit
//! `/jobs/:id/download`. An undeclared path 404s at Core in a way that reads exactly
//! like a router bug in this file, which is why the two lists are kept adjacent in
//! review.
//!
//! Error mapping is deliberate rather than uniform: picking a file the app cannot
//! read (415), naming a language it does not know (400) and pointing at a file that
//! has since moved (404) are three different things the companion says three
//! different ways, and collapsing them into 500 would make all three read as "the
//! app is broken".

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Notify;

use crate::cues::{Format, Layout};
use crate::engine;
use crate::languages;
use crate::library::{self, PathError};
use crate::store::{Job, NewJob, Status, SubtitleStore};

/// Everything a handler needs. Cheap to clone.
#[derive(Clone)]
pub struct Ctx {
    pub store: SubtitleStore,
    /// Notified when a job is created, so the worker starts it immediately.
    pub wake: Arc<Notify>,
    pub events: Arc<ryu_app_events::EventEmitter>,
}

/// How many jobs the list view returns. A node that has subtitled a thousand files
/// does not need all of them on first paint.
const LIST_LIMIT: usize = 200;

/// The router, state baked in so the caller nests a `Router<()>`.
pub fn routes(ctx: Ctx) -> Router {
    Router::new()
        .route("/languages", get(list_languages))
        .route("/roots", get(list_roots))
        .route("/library", get(browse))
        .route("/settings", get(get_settings).put(put_settings))
        .route("/jobs", get(list_jobs).post(create_job))
        .route("/jobs/:id", get(get_job).delete(delete_job))
        .route("/jobs/:id/cues", get(get_cues))
        .route("/jobs/:id/download", get(download))
        .route("/jobs/:id/retry", post(retry_job))
        .route("/jobs/:id/cancel", post(cancel_job))
        .with_state(ctx)
}

/// The OpenAPI sub-document Core fetches from `GET /openapi.json` and lowers into
/// one LLM tool per operation.
///
/// Deriving tools from this document is the ONLY path an agent has into this app,
/// so an unannotated route is not "undocumented" — it is uncallable. Core also
/// INTERSECTS the operations against `sidecars[0].http.routes[]`, so an operation
/// documented here but absent from the manifest yields nothing at all.
pub fn openapi() -> utoipa::openapi::OpenApi {
    <SubtitlesApiDoc as utoipa::OpenApi>::openapi()
}

/// The document itself.
///
/// `components(schemas(...))` is what makes each `request_body = T` resolve to a real
/// `#/components/schemas/T`. Without the entry the operation still carries a `$ref`,
/// but Core's `resolve_ref` finds nothing and derives a write tool with ZERO visible
/// arguments — discoverable and uncallable, which is worse than absent. utoipa 5 also
/// auto-collects schemas reachable from the annotated paths; the rows are listed
/// anyway so the registration is greppable and cannot be lost to an attribute edit.
#[derive(utoipa::OpenApi)]
#[openapi(
    paths(
        browse,
        cancel_job,
        create_job,
        delete_job,
        download,
        get_cues,
        get_job,
        get_settings,
        list_jobs,
        list_languages,
        list_roots,
        put_settings,
        retry_job,
    ),
    components(schemas(CreateJobInput, SettingsInput))
)]
struct SubtitlesApiDoc;

/// `GET /languages` — the closed target-language table the picker renders.
#[utoipa::path(
    get,
    path = "/api/subtitles/languages",
    tag = "Subtitles",
    summary = "List the target languages a subtitle job can translate into.",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn list_languages() -> impl IntoResponse {
    Json(json!({ "languages": languages::LANGUAGES }))
}

/// `GET /roots` — the browsable starting points (Movies, Downloads, …).
#[utoipa::path(
    get,
    path = "/api/subtitles/roots",
    tag = "Subtitles",
    summary = "List the media folders this app is allowed to browse. Call this first — every other path argument must sit under one of these roots.",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn list_roots() -> impl IntoResponse {
    Json(json!({ "roots": library::roots() }))
}

#[derive(Debug, Deserialize)]
struct BrowseQuery {
    /// Absolute directory to list. Omitted means "the roots" — the companion asks
    /// for those separately, so this is only a convenience.
    dir: Option<String>,
}

/// `GET /library?dir=…` — one directory of subfolders and media files.
#[utoipa::path(
    get,
    path = "/api/subtitles/library",
    tag = "Subtitles",
    summary = "List the sub-folders and video files inside one media folder, to find a video to subtitle.",
    params((
        "dir" = Option<String>, Query,
        // Spelling out the roots constraint is not padding: `validate_source` rejects
        // anything outside them with a 403, and a model told only "absolute path"
        // reliably invents `/etc` or a home directory that is not a declared root.
        description = "Absolute path of the folder to list. It must be one of the roots from GET /api/subtitles/roots, or a folder beneath one — anything else is refused. Omit to get the roots themselves."
    )),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn browse(Query(query): Query<BrowseQuery>) -> Response {
    let Some(dir) = query.dir.filter(|d| !d.trim().is_empty()) else {
        return Json(json!({ "entries": [], "roots": library::roots() })).into_response();
    };
    match library::list_dir(&PathBuf::from(dir.trim())) {
        Ok(entries) => Json(json!({ "entries": entries })).into_response(),
        Err(e) => path_error(&e),
    }
}

/// `GET /settings` — node defaults for new jobs.
#[utoipa::path(
    get,
    path = "/api/subtitles/settings",
    tag = "Subtitles",
    summary = "Read the node-wide defaults a new subtitle job inherits (target language, file format, layout, engine, model).",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn get_settings(State(ctx): State<Ctx>) -> Response {
    match ctx.store.settings().await {
        Ok(settings) => Json(settings).into_response(),
        Err(e) => server_error(&e.to_string()),
    }
}

/// Partial update of the node defaults. Every field is optional; an omitted field
/// keeps its current value rather than resetting it.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct SettingsInput {
    /// Default target language as a BCP-47 code from `GET /api/subtitles/languages`
    /// (e.g. `es`, `pt-BR`). An unknown code is rejected with 400.
    target_language: Option<String>,
    /// Default subtitle file format: `srt` or `vtt`.
    format: Option<String>,
    /// Default cue layout: `translated` (target language only), `source` (original
    /// only), or `bilingual` (both stacked in each cue).
    layout: Option<String>,
    /// Default speech-to-text engine. `whisper` is the only engine that returns
    /// per-cue timings, so leave it alone unless you have a reason.
    engine: Option<String>,
    /// Default model id used for the translation pass.
    model: Option<String>,
    /// Whether a finished job also writes the subtitle file next to the source video
    /// instead of keeping it only inside the app.
    write_beside_source: Option<bool>,
}

/// `PUT /settings` — partial update; anything omitted keeps its current value.
#[utoipa::path(
    put,
    path = "/api/subtitles/settings",
    tag = "Subtitles",
    summary = "Change the node-wide defaults new subtitle jobs inherit. Does not affect jobs that already exist.",
    request_body = SettingsInput,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn put_settings(State(ctx): State<Ctx>, Json(input): Json<SettingsInput>) -> Response {
    let mut settings = ctx.store.settings().await.unwrap_or_default();
    if let Some(code) = input.target_language {
        let Some(language) = languages::find(&code) else {
            return bad_request(&format!("unknown target language `{code}`"));
        };
        settings.target_language = language.code.to_string();
    }
    if let Some(format) = input.format {
        let Some(parsed) = Format::parse(&format) else {
            return bad_request(&format!("unknown subtitle format `{format}`"));
        };
        settings.format = parsed;
    }
    if let Some(layout) = input.layout {
        let Some(parsed) = parse_layout(&layout) else {
            return bad_request(&format!("unknown layout `{layout}`"));
        };
        settings.layout = parsed;
    }
    if let Some(engine) = input.engine {
        let engine = engine.trim().to_string();
        if !engine.is_empty() {
            settings.engine = engine;
        }
    }
    if let Some(model) = input.model {
        settings.model = model.trim().to_string();
    }
    if let Some(beside) = input.write_beside_source {
        settings.write_beside_source = beside;
    }
    match ctx.store.save_settings(&settings).await {
        Ok(()) => Json(settings).into_response(),
        Err(e) => server_error(&e.to_string()),
    }
}

/// `GET /jobs` — newest first, with a `source_missing` flag so the UI can grey out
/// a job whose video has been moved or deleted.
#[utoipa::path(
    get,
    path = "/api/subtitles/jobs",
    tag = "Subtitles",
    summary = "List subtitle jobs, newest first, with their status and progress.",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn list_jobs(State(ctx): State<Ctx>) -> Response {
    match ctx.store.list_jobs(LIST_LIMIT).await {
        Ok(jobs) => {
            Json(json!({ "jobs": jobs.iter().map(decorate).collect::<Vec<_>>() })).into_response()
        }
        Err(e) => server_error(&e.to_string()),
    }
}

/// What a new subtitle job needs. Only `source_path` is required; every override
/// falls back to the node default from `GET /api/subtitles/settings`.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct CreateJobInput {
    /// Absolute path of the video file to subtitle. It must be a media file under one
    /// of the roots from `GET /api/subtitles/roots`; get it from
    /// `GET /api/subtitles/library` rather than guessing.
    source_path: String,
    /// BCP-47 code of the language to translate into, from
    /// `GET /api/subtitles/languages`. Omit to use the node default.
    target_language: Option<String>,
    /// Subtitle file format: `srt` or `vtt`. Omit to use the node default.
    format: Option<String>,
    /// Cue layout: `translated`, `source`, or `bilingual`. Omit to use the node
    /// default.
    layout: Option<String>,
    /// Speech-to-text engine override. Whisper is the default and the only engine
    /// that returns per-cue timings, so an override usually makes the output worse.
    engine: Option<String>,
    /// Model id override for the translation pass. Omit to use the node default.
    model: Option<String>,
}

/// `POST /jobs` — validate, queue, and wake the worker.
///
/// Returns immediately with a queued job: transcribing a film is minutes of work,
/// and a request that waited for it would die on the proxy timeout long before the
/// file existed.
#[utoipa::path(
    post,
    path = "/api/subtitles/jobs",
    tag = "Subtitles",
    summary = "Queue a new subtitle job for one video file: transcribe it, translate the transcript, and write a timed subtitle file. Returns immediately — the job runs for minutes, so poll GET /api/subtitles/jobs/{id} for progress.",
    request_body = CreateJobInput,
    responses((status = 201, description = "Queued", body = serde_json::Value))
)]
async fn create_job(State(ctx): State<Ctx>, Json(input): Json<CreateJobInput>) -> Response {
    let source = match library::validate_source(&input.source_path) {
        Ok(path) => path,
        Err(e) => return path_error(&e),
    };

    let settings = ctx.store.settings().await.unwrap_or_default();
    let code = input
        .target_language
        .unwrap_or_else(|| settings.target_language.clone());
    let Some(language) = languages::find(&code) else {
        return bad_request(&format!("unknown target language `{code}`"));
    };
    let format = match input.format {
        Some(f) => match Format::parse(&f) {
            Some(parsed) => parsed,
            None => return bad_request(&format!("unknown subtitle format `{f}`")),
        },
        None => settings.format,
    };
    let layout = match input.layout {
        Some(l) => match parse_layout(&l) {
            Some(parsed) => parsed,
            None => return bad_request(&format!("unknown layout `{l}`")),
        },
        None => settings.layout,
    };

    let new = NewJob {
        source_name: source
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "video".to_string()),
        source_path: source.to_string_lossy().into_owned(),
        target_language: language.code.to_string(),
        format,
        layout,
        engine: input
            .engine
            .map(|e| e.trim().to_string())
            .filter(|e| !e.is_empty())
            .unwrap_or_else(|| settings.engine.clone()),
        model: input
            .model
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| settings.model.clone()),
    };

    match ctx.store.create_job(new).await {
        Ok(job) => {
            ctx.wake.notify_one();
            ctx.events
                .emit(
                    "@ryu/subtitles#job.queued",
                    json!({
                        "job_id": job.id,
                        "source_name": job.source_name,
                        "target_language": job.target_language,
                    }),
                )
                .await;
            (StatusCode::CREATED, Json(decorate(&job))).into_response()
        }
        Err(e) => server_error(&e.to_string()),
    }
}

/// `GET /jobs/:id` — the poll endpoint the companion drives its progress bar from.
#[utoipa::path(
    get,
    path = "/api/subtitles/jobs/{id}",
    tag = "Subtitles",
    summary = "Read one subtitle job: its status, progress, and whether the finished file can be downloaded yet.",
    params(("id" = String, Path, description = "Job id, as returned by POST or GET /api/subtitles/jobs")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn get_job(State(ctx): State<Ctx>, Path(id): Path<String>) -> Response {
    match ctx.store.get_job(&id).await {
        Ok(Some(job)) => Json(decorate(&job)).into_response(),
        Ok(None) => not_found("no such job"),
        Err(e) => server_error(&e.to_string()),
    }
}

/// `GET /jobs/:id/cues` — the cue list, for the in-app transcript view.
#[utoipa::path(
    get,
    path = "/api/subtitles/jobs/{id}/cues",
    tag = "Subtitles",
    summary = "Read the timed transcript of a finished job as structured cues — use this to read or quote what was said, rather than downloading the file.",
    params(("id" = String, Path, description = "Job id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn get_cues(State(ctx): State<Ctx>, Path(id): Path<String>) -> Response {
    match ctx.store.get_job(&id).await {
        Ok(Some(_)) => match ctx.store.cues(&id).await {
            Ok(cues) => Json(json!({ "cues": cues })).into_response(),
            Err(e) => server_error(&e.to_string()),
        },
        Ok(None) => not_found("no such job"),
        Err(e) => server_error(&e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct DownloadQuery {
    /// Render as this format instead of the job's. Re-rendering is free — the cues
    /// are stored — so switching SRT↔WebVTT never re-transcribes.
    format: Option<String>,
    /// Render this layout instead of the job's (target-only / source / bilingual).
    layout: Option<String>,
}

/// `GET /jobs/:id/download` — the subtitle file itself.
#[utoipa::path(
    get,
    path = "/api/subtitles/jobs/{id}/download",
    tag = "Subtitles",
    summary = "Download the finished subtitle file as text. Re-rendering into another format or layout is free and never re-transcribes.",
    params(
        ("id" = String, Path, description = "Job id"),
        ("format" = Option<String>, Query, description = "Render as `srt` or `vtt` instead of the job's own format."),
        ("layout" = Option<String>, Query, description = "Render as `translated`, `source`, or `bilingual` instead of the job's own layout.")
    ),
    responses((status = 200, description = "The subtitle file", body = String))
)]
async fn download(
    State(ctx): State<Ctx>,
    Path(id): Path<String>,
    Query(query): Query<DownloadQuery>,
) -> Response {
    let job = match ctx.store.get_job(&id).await {
        Ok(Some(job)) => job,
        Ok(None) => return not_found("no such job"),
        Err(e) => return server_error(&e.to_string()),
    };
    let format = match query.format {
        Some(f) => match Format::parse(&f) {
            Some(parsed) => parsed,
            None => return bad_request(&format!("unknown subtitle format `{f}`")),
        },
        None => job.format,
    };
    let layout = match query.layout {
        Some(l) => match parse_layout(&l) {
            Some(parsed) => parsed,
            None => return bad_request(&format!("unknown layout `{l}`")),
        },
        None => job.layout,
    };

    let cues = match ctx.store.cues(&id).await {
        Ok(cues) => cues,
        Err(e) => return server_error(&e.to_string()),
    };
    if cues.is_empty() {
        return (
            StatusCode::CONFLICT,
            Json(json!({ "error": "this job has no subtitles yet" })),
        )
            .into_response();
    }

    // Remember the choice, so the next download (and any "open the file" affordance)
    // agrees with what was just downloaded.
    if format != job.format || layout != job.layout {
        let _ = ctx.store.set_render_options(&id, format, layout).await;
    }

    let body = crate::cues::render(&cues, format, layout);
    let filename = download_filename(&job, format);
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, format.content_type().to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{filename}\""),
            ),
        ],
        body,
    )
        .into_response()
}

/// `POST /jobs/:id/retry` — re-queue a finished or failed job.
// No `request_body`: the handler takes only `State` + `Path`, and declaring an
// untyped body would give the derived tool a phantom argument.
#[utoipa::path(
    post,
    path = "/api/subtitles/jobs/{id}/retry",
    tag = "Subtitles",
    summary = "Re-run a finished or failed subtitle job from the start, discarding its previous output. Rejected with 409 while the job is still running.",
    params(("id" = String, Path, description = "Job id")),
    responses((status = 200, description = "Re-queued", body = serde_json::Value))
)]
async fn retry_job(State(ctx): State<Ctx>, Path(id): Path<String>) -> Response {
    match ctx.store.get_job(&id).await {
        Ok(Some(job)) if job.status.is_terminal() => {
            if let Err(e) = ctx.store.requeue(&id).await {
                return server_error(&e.to_string());
            }
            engine::remove_data_output(&id);
            ctx.wake.notify_one();
            match ctx.store.get_job(&id).await {
                Ok(Some(job)) => Json(decorate(&job)).into_response(),
                _ => server_error("the job vanished while being re-queued"),
            }
        }
        Ok(Some(_)) => (
            StatusCode::CONFLICT,
            Json(json!({ "error": "this job is still running" })),
        )
            .into_response(),
        Ok(None) => not_found("no such job"),
        Err(e) => server_error(&e.to_string()),
    }
}

/// `POST /jobs/:id/cancel` — stop a running job at its next window boundary.
#[utoipa::path(
    post,
    path = "/api/subtitles/jobs/{id}/cancel",
    tag = "Subtitles",
    summary = "Stop a running subtitle job at its next window boundary. Already-finished jobs are returned unchanged.",
    params(("id" = String, Path, description = "Job id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn cancel_job(State(ctx): State<Ctx>, Path(id): Path<String>) -> Response {
    match ctx.store.get_job(&id).await {
        Ok(Some(job)) if !job.status.is_terminal() => {
            match ctx.store.cancel_if_active(&id, "Cancelled.").await {
                Ok(_) => match ctx.store.get_job(&id).await {
                    Ok(Some(job)) => Json(decorate(&job)).into_response(),
                    _ => server_error("the job vanished while being cancelled"),
                },
                Err(e) => server_error(&e.to_string()),
            }
        }
        Ok(Some(job)) => Json(decorate(&job)).into_response(),
        Ok(None) => not_found("no such job"),
        Err(e) => server_error(&e.to_string()),
    }
}

/// `DELETE /jobs/:id` — remove the job and its generated file. The SOURCE VIDEO is
/// never touched, and neither is a subtitle file written beside it: the user asked
/// to forget a job, not to delete the artifact they may already be using.
#[utoipa::path(
    delete,
    path = "/api/subtitles/jobs/{id}",
    tag = "Subtitles",
    summary = "Forget a subtitle job and its in-app output. The source video, and any subtitle file already written beside it, are left untouched.",
    params(("id" = String, Path, description = "Job id")),
    responses((status = 204, description = "Deleted"))
)]
async fn delete_job(State(ctx): State<Ctx>, Path(id): Path<String>) -> Response {
    match ctx.store.delete_job(&id).await {
        Ok(true) => {
            engine::remove_data_output(&id);
            StatusCode::NO_CONTENT.into_response()
        }
        Ok(false) => not_found("no such job"),
        Err(e) => server_error(&e.to_string()),
    }
}

/// A job plus the two derived flags the UI needs and the row does not carry.
fn decorate(job: &Job) -> serde_json::Value {
    let mut value = serde_json::to_value(job).unwrap_or_else(|_| json!({}));
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "source_missing".to_string(),
            json!(!engine::source_exists(&job.source_path)),
        );
        object.insert(
            "downloadable".to_string(),
            json!(job.status == Status::Completed && job.cue_count > 0),
        );
        object.insert(
            "language_name".to_string(),
            json!(languages::find(&job.target_language).map(|l| l.name)),
        );
    }
    value
}

/// `<video name>.<lang>.<ext>` — the same convention as the file written beside the
/// source, so a user who downloads it and drops it next to the video gets a track
/// their player picks up.
fn download_filename(job: &Job, format: Format) -> String {
    let stem = std::path::Path::new(&job.source_name)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "subtitles".to_string());
    // Quotes and control characters would break out of the Content-Disposition
    // header's quoted-string; the source name comes from the filesystem, so it is
    // not trusted to be header-safe.
    let stem: String = stem
        .chars()
        .filter(|c| !c.is_control() && *c != '"' && *c != '\\')
        .collect();
    format!("{stem}.{}.{}", job.target_language, format.extension())
}

fn parse_layout(value: &str) -> Option<Layout> {
    match value.trim().to_ascii_lowercase().as_str() {
        "translated" | "target" => Some(Layout::Translated),
        "source" | "original" => Some(Layout::Source),
        "bilingual" | "both" | "dual" => Some(Layout::Bilingual),
        _ => None,
    }
}

fn path_error(e: &PathError) -> Response {
    let status = match e {
        PathError::OutsideRoots => StatusCode::FORBIDDEN,
        PathError::Missing => StatusCode::NOT_FOUND,
        PathError::NotMedia => StatusCode::UNSUPPORTED_MEDIA_TYPE,
    };
    (status, Json(json!({ "error": e.to_string() }))).into_response()
}

fn bad_request(message: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": message }))).into_response()
}

fn not_found(message: &str) -> Response {
    (StatusCode::NOT_FOUND, Json(json!({ "error": message }))).into_response()
}

fn server_error(message: &str) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": message })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Job;

    #[test]
    fn multi_method_paths_keep_every_operation() {
        // utoipa keys `paths` by path STRING, so handlers annotated separately on the
        // same path must MERGE into one PathItem. If one overwrote another, the path key
        // would still exist and the write body would still resolve — the read tool would
        // silently never exist, which is exactly the failure this document prevents. The
        // route-coverage test above cannot see that, because it only checks the key.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        for (path, methods) in [
            ("/api/subtitles/settings", &["get", "put"][..]),
            ("/api/subtitles/jobs", &["get", "post"][..]),
            ("/api/subtitles/jobs/{id}", &["get", "delete"][..]),
        ] {
            let item = wire
                .pointer(&format!("/paths/{}", path.replace('/', "~1")))
                .unwrap_or_else(|| panic!("{path} has no PathItem"));
            for method in methods {
                assert!(
                    item.get(method).is_some(),
                    "{path} lost its {method} operation"
                );
            }
        }
    }

    // ── OpenAPI document ───────────────────────────────────────────────────────

    /// This app's own manifest, read at compile time. The route contract lives there,
    /// so the invariants below compare the document against the real declaration
    /// rather than against a second list that could drift from it.
    fn openapi_manifest() -> serde_json::Value {
        serde_json::from_str(include_str!("../../manifest.json")).expect("valid JSON")
    }

    /// The manifest sidecar whose HTTP surface this router serves: the one declaring an
    /// `http.mount`. Selected BY mount rather than by index so that adding a second,
    /// mountless sidecar later cannot silently start asserting against the wrong process.
    fn mounted_sidecar() -> serde_json::Value {
        openapi_manifest()["sidecars"]
            .as_array()
            .expect("sidecars must be an array")
            .iter()
            .find(|s| s["http"]["mount"].is_string())
            .expect("one sidecar must declare an http.mount")
            .clone()
    }

    /// A manifest route (relative to the mount, in axum's `:param` form) rewritten into
    /// the form the OpenAPI document uses (absolute, in `{param}` form).
    ///
    /// The two forms differ ON PURPOSE — the router registers paths relative to the
    /// mount because Core nests it there, while `#[utoipa::path]` carries the absolute
    /// EXTERNAL path a caller actually hits. Normalise here; do not "align" either side.
    fn doc_path_for(mount: &str, route: &str) -> String {
        let joined = if route == "/" {
            mount.to_owned()
        } else {
            format!("{mount}{route}")
        };
        joined
            .split('/')
            .map(|seg| match seg.strip_prefix(':') {
                Some(name) => format!("{{{name}}}"),
                None => seg.to_owned(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    #[test]
    fn openapi_doc_covers_the_served_routes() {
        assert!(!super::openapi().paths.paths.is_empty());
    }

    #[test]
    fn every_declared_route_appears_in_the_openapi_doc() {
        // The direction that decides tool yield. Core's importer keeps only the document
        // operations the manifest ALSO declares, so a declared route with no
        // `#[utoipa::path]` is a tool that silently never exists — nothing errors, the
        // agent simply cannot call it. (The other direction is harmless: an annotated
        // path the manifest does not declare is dropped by the same filter.)
        let sidecar = mounted_sidecar();
        let mount = sidecar["http"]["mount"].as_str().expect("an http.mount");
        let doc = super::openapi();
        for route in sidecar["http"]["routes"]
            .as_array()
            .expect("routes must be an array")
        {
            let path = route["path"].as_str().expect("a route path");
            let expected = doc_path_for(mount, path);
            assert!(
                doc.paths.paths.contains_key(&expected),
                "'{path}' is declared in manifest.json but the OpenAPI document has no \
                 '{expected}' operation — Core derives no tool for it"
            );
        }
    }

    #[test]
    fn write_operations_carry_a_typed_request_body() {
        // An untyped body (`serde_json::Value`) still produces an operation, so the tool
        // is DISCOVERABLE — with zero visible arguments, hence uncallable. Assert the
        // `$ref` resolves into `components.schemas` the way Core's `resolve_ref` will.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        for (path, method) in [
            ("/api/subtitles/settings", "put"),
            ("/api/subtitles/jobs", "post"),
        ] {
            let schema = wire
                .pointer(&format!(
                    "/paths/{}/{method}/requestBody/content/application~1json/schema/$ref",
                    path.replace('/', "~1")
                ))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| panic!("{method} {path} must declare a typed request body"));
            let name = schema
                .rsplit('/')
                .next()
                .expect("a $ref always has a last segment");
            assert!(
                wire.pointer(&format!("/components/schemas/{name}"))
                    .is_some(),
                "{method} {path} references {schema}, which is missing from components.schemas"
            );
        }
    }

    #[test]
    fn id_only_operations_declare_no_body() {
        // Retry/cancel take only `State` + `Path`. A `request_body` here would invent an
        // argument the handler never reads, which a model then tries to fill.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        for path in [
            "/api/subtitles/jobs/{id}/retry",
            "/api/subtitles/jobs/{id}/cancel",
        ] {
            let op = wire
                .pointer(&format!("/paths/{}/post", path.replace('/', "~1")))
                .unwrap_or_else(|| panic!("{path} must have a POST operation"));
            assert!(
                op.get("requestBody").is_none(),
                "{path} takes no body but the document declares one"
            );
            assert!(
                op.get("parameters").is_some(),
                "{path} must still document its path id"
            );
        }
    }

    fn job() -> Job {
        Job {
            id: "sub_1".into(),
            source_path: "/movies/The Film.mkv".into(),
            source_name: "The Film.mkv".into(),
            target_language: "es".into(),
            format: Format::Srt,
            layout: Layout::Translated,
            status: Status::Completed,
            progress: 1.0,
            stage: "Done".into(),
            error: None,
            duration_ms: Some(1000),
            cue_count: 12,
            translated_count: 12,
            engine: "whisper".into(),
            model: String::new(),
            output_path: None,
            created_at: 0,
            updated_at: 0,
            completed_at: Some(0),
        }
    }

    #[test]
    fn download_filename_follows_the_player_convention() {
        assert_eq!(download_filename(&job(), Format::Srt), "The Film.es.srt");
        assert_eq!(download_filename(&job(), Format::Vtt), "The Film.es.vtt");
    }

    #[test]
    fn download_filename_cannot_break_the_content_disposition_header() {
        let mut j = job();
        j.source_name = "we\"ird\r\nname.mkv".into();
        let name = download_filename(&j, Format::Srt);
        assert!(!name.contains('"'));
        assert!(!name.contains('\n') && !name.contains('\r'));
    }

    #[test]
    fn layout_parsing_accepts_the_aliases_the_ui_sends() {
        assert_eq!(parse_layout("Bilingual"), Some(Layout::Bilingual));
        assert_eq!(parse_layout("both"), Some(Layout::Bilingual));
        assert_eq!(parse_layout("original"), Some(Layout::Source));
        assert_eq!(parse_layout("target"), Some(Layout::Translated));
        assert_eq!(parse_layout("sideways"), None);
    }

    #[test]
    fn decoration_reports_a_missing_source_and_download_readiness() {
        let value = decorate(&job());
        assert_eq!(value["source_missing"], json!(true), "/movies is fictional");
        assert_eq!(value["downloadable"], json!(true));
        assert_eq!(value["language_name"], json!("Spanish"));

        let mut running = job();
        running.status = Status::Transcribing;
        running.cue_count = 0;
        assert_eq!(decorate(&running)["downloadable"], json!(false));
    }

    #[test]
    fn path_errors_map_to_three_distinguishable_statuses() {
        assert_eq!(
            path_error(&PathError::Missing).status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            path_error(&PathError::OutsideRoots).status(),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            path_error(&PathError::NotMedia).status(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE
        );
    }
}
