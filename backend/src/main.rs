//! `ryu-subtitles` — the standalone, out-of-process Subtitles sidecar.
//!
//! Pick a video on this machine, transcribe it locally, translate the transcript
//! into a chosen language, and write a timed `.srt`/`.vtt` beside the file. It runs
//! as a SEPARATE PROCESS Core spawns, health-checks, and proxies to on loopback —
//! the same shape as `ryu-social` / `ryu-mail`. Core does NOT contain this code and
//! does not link it: there is no `lib.rs`, every module below is bin-private, and
//! the only route into this process is the generic ext-proxy. So subtitling scales,
//! fails, and ships independently of the rest of the node.
//!
//! Contract surface — the paths Core forwards to, byte-identical whether they arrive
//! via the `public_mount` (`/api/subtitles/*`) or the plugin proxy
//! (`/api/ext/@ryu/subtitles/*`, rewritten onto the same prefix):
//!
//! ```text
//!   /health                        — un-gated loopback probe
//!   /api/subtitles/*               — the whole app surface (see `api::routes`)
//! ```
//!
//! # Everything local
//!
//! Both model hops stay on the machine by default. Transcription goes through the
//! extracted [`ryu_stt`] crate to local whisper.cpp; translation goes to the LOCAL
//! gateway (`127.0.0.1:7981`) with the bundled on-device model. Nothing here names a
//! remote provider — a node with no keys configured produces subtitles anyway, and
//! the media never leaves the disk it is already on.
//!
//! # Why whisper, specifically
//!
//! `ryu-stt` is built at DEFAULT features (no `voice-parakeet`), so
//! `default_stt_engine()` resolves to whisper. That is not a lean-build accident: a
//! subtitle file IS its timings, and whisper is the engine that returns per-segment
//! `start_ms`/`end_ms` (from `verbose_json`). Parakeet returns text only — its
//! `segments` is always empty — so a parakeet job degrades to one coarse cue per
//! 30-second window rather than real subtitles.
//!
//! SECURITY: this binary binds LOOPBACK ONLY (127.0.0.1) **and** guards every
//! `/api/subtitles/*` route with a shared-secret bearer (`RYU_EXT_TOKEN`, injected by
//! Core into this child's spawn env). Core stays the auth front — it runs its own
//! `require_auth`, then re-stamps `Authorization: Bearer <RYU_EXT_TOKEN>` on the
//! loopback hop — so a request that did NOT come through Core (any other local
//! process on a shared host) is rejected with 401. The gate is FAIL-CLOSED: with no
//! token configured, every protected route rejects rather than falling open. That
//! matters more here than in a CRUD app: this process reads files off the user's
//! disk by path, so an un-gated route would be a file-reading primitive for anything
//! else running locally. The path gate in `library.rs` is the second lock.
//!
//! Port: `RYU_SUBTITLES_PORT` env, default `8013`. Data dir: resolved via the
//! inlined `paths::ryu_dir` (`RYU_DIR`-env-first, injected by Core at spawn), so it
//! opens the SAME `subtitles.db` the node uses and writes generated files under the
//! SAME `subtitles/` directory the app downloads from.

mod api;
mod cues;
mod engine;
mod languages;
mod library;
mod media;
mod paths;
mod store;
mod translate;

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use axum::{
    extract::Request,
    http::{header::AUTHORIZATION, StatusCode},
    middleware::{from_fn, Next},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::json;
use tokio::sync::Notify;

use crate::store::SubtitleStore;

/// Default loopback port for the subtitles sidecar (overridable via
/// `RYU_SUBTITLES_PORT`, which Core injects profile-shifted so concurrent
/// dev/release nodes do not collide on it).
///
/// Must stay equal to `sidecars[0].port` in `apps-store/subtitles/manifest.json`: the
/// manifest value is what Core injects and what its health probe polls, and this
/// constant is only the standalone-run fallback — a drift between the two is a
/// sidecar that Core reports unhealthy while it happily serves on another port.
/// There is no port registry (see `SidecarSpec::port`), so avoiding a collision is
/// this file's job: 7990–8012 are taken by the existing apps-store sidecars.
const DEFAULT_PORT: u16 = 8013;

/// The external prefix. Must match the manifest's `sidecars[0].http.mount`, or Core
/// will forward `/api/subtitles/jobs` to a router that only knows `/jobs`.
const MOUNT: &str = "/api/subtitles";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let port: u16 = std::env::var("RYU_SUBTITLES_PORT")
        .ok()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(DEFAULT_PORT);

    // The shared secret Core injects via the generic ext-proxy loader: a per-plugin
    // minted token it stamps on every proxied hop and on the health probe.
    let token = std::env::var("RYU_EXT_TOKEN")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
    if token.is_some() {
        tracing::info!(
            "ryu-subtitles: protected {MOUNT}/* routes require the injected shared-secret bearer"
        );
    } else {
        tracing::warn!(
            "ryu-subtitles: no RYU_EXT_TOKEN set; protected {MOUNT}/* routes are FAIL-CLOSED (reject all). Core injects this token when it spawns the sidecar."
        );
    }

    let store = SubtitleStore::open(paths::ryu_dir().join("subtitles.db"))?;

    // One HTTP client for the whole process: the STT hop and the translation hop
    // share its connection pool. The timeout is generous because a local whisper.cpp
    // pass over a 30-second window on a cold model can genuinely take a minute, and
    // a premature timeout would fail a job that was about to succeed.
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .unwrap_or_default();

    let events = Arc::new(ryu_app_events::EventEmitter::with_client(
        engine::PLUGIN_ID,
        http.clone(),
    ));
    let wake = Arc::new(Notify::new());

    let worker = engine::spawn(engine::Worker {
        store: store.clone(),
        http: http.clone(),
        events: events.clone(),
        wake: wake.clone(),
    });

    let ctx = api::Ctx {
        store: store.clone(),
        wake,
        events,
    };

    // The app router, with the shared-secret gate layered over the WHOLE nest. There
    // is no public route here: nothing external calls this app, so every path under
    // the mount is protected without exception.
    let gated_token = token.clone();
    let app_routes = Router::new()
        .nest(MOUNT, api::routes(ctx))
        .layer(from_fn(move |req: Request, next: Next| {
            let expected = gated_token.clone();
            async move { require_token(req, next, expected.as_deref()).await }
        }));

    // `/health` sits OUTSIDE the gated nest so Core's loopback probe succeeds before
    // auth. It asserts the DB is READABLE (not merely that the process is alive) and
    // returns no user data.
    let health_store = store;
    let app = Router::new()
        .route(
            "/health",
            get(move || {
                let store = health_store.clone();
                async move { health(store).await }
            }),
        )
        .merge(app_routes);

    // LOOPBACK ONLY (belt) + shared-secret bearer (suspenders).
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("ryu-subtitles sidecar listening on http://{addr}{MOUNT}");

    let result = axum::serve(listener, app).await;
    // Stop the queue on shutdown so a supervised restart never runs two workers over
    // one database — which would transcribe the same job twice, in parallel, at half
    // speed each.
    worker.abort();
    result?;
    Ok(())
}

/// Un-gated loopback health probe. Confirms DB readiness with a cheap read and
/// returns counts only — never content.
async fn health(store: SubtitleStore) -> Response {
    match store.job_count().await {
        Ok(jobs) => (StatusCode::OK, Json(json!({ "ok": true, "jobs": jobs }))).into_response(),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "ok": false, "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// Shared-secret bearer gate for the proxied surface.
///
/// **Fail-closed:** `expected == None`/empty (no token configured) rejects every
/// request rather than falling open, so a bare-run or misconfigured sidecar never
/// reads a user's files for whatever else is on the machine.
async fn require_token(req: Request, next: Next, expected: Option<&str>) -> Response {
    let provided = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    if bearer_ok(provided, expected) {
        next.run(req).await
    } else {
        (StatusCode::UNAUTHORIZED, "unauthorized").into_response()
    }
}

/// Pure bearer check, factored out so the auth decision is unit-testable without an
/// axum `Request`/`Next`. Returns `true` only when `expected` is a non-empty token
/// AND `provided` equals it (constant-time compared).
fn bearer_ok(provided: Option<&str>, expected: Option<&str>) -> bool {
    let Some(expected) = expected.filter(|t| !t.is_empty()) else {
        return false;
    };
    ct_eq(provided.unwrap_or("").as_bytes(), expected.as_bytes())
}

/// Constant-time byte comparison — no early return on the first mismatched byte, so
/// the token check does not leak length/prefix via timing.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::{bearer_ok, DEFAULT_PORT, MOUNT};

    #[test]
    fn bearer_ok_matches_only_exact_nonempty_token() {
        assert!(bearer_ok(Some("secret"), Some("secret")));
        assert!(!bearer_ok(Some("secret"), Some("other")));
        assert!(!bearer_ok(Some("secre"), Some("secret")));
        assert!(!bearer_ok(None, Some("secret")));
    }

    #[test]
    fn the_gate_is_fail_closed_with_no_token_configured() {
        assert!(!bearer_ok(Some("anything"), None));
        assert!(!bearer_ok(None, None));
        assert!(!bearer_ok(Some(""), Some("")));
    }

    /// The two constants Core's manifest must agree with. A drift in either is a
    /// sidecar Core reports unhealthy, or routes it forwards to a 404.
    #[test]
    fn port_and_mount_match_the_manifest() {
        let manifest = include_str!("../../manifest.json");
        let value: serde_json::Value = serde_json::from_str(manifest).expect("manifest parses");
        let sidecar = &value["sidecars"][0];
        assert_eq!(sidecar["port"].as_u64(), Some(u64::from(DEFAULT_PORT)));
        assert_eq!(sidecar["http"]["mount"].as_str(), Some(MOUNT));
        assert_eq!(sidecar["http"]["public_mount"].as_str(), Some(MOUNT));
    }

    /// Core matches `http.routes[]` as an ALLOWLIST with exact segment counts, so a
    /// route this binary serves but the manifest omits is a hard 404 that reads like
    /// a router bug. This asserts the two lists agree.
    #[test]
    fn every_served_route_is_declared_in_the_manifest() {
        let manifest = include_str!("../../manifest.json");
        let value: serde_json::Value = serde_json::from_str(manifest).expect("manifest parses");
        let declared: Vec<String> = value["sidecars"][0]["http"]["routes"]
            .as_array()
            .expect("routes array")
            .iter()
            .filter_map(|r| r["path"].as_str().map(std::string::ToString::to_string))
            .collect();
        for served in [
            "/languages",
            "/roots",
            "/library",
            "/settings",
            "/jobs",
            "/jobs/:id",
            "/jobs/:id/cues",
            "/jobs/:id/download",
            "/jobs/:id/retry",
            "/jobs/:id/cancel",
        ] {
            assert!(
                declared.iter().any(|d| d == served),
                "`{served}` is served but not declared in the manifest's http.routes"
            );
        }
    }
}
