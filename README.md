# ryu-subtitles

Subtitles for Ryu — pick a video on your own machine, transcribe it with local whisper, translate the transcript into the language you want, and write a timed .srt/.vtt beside the file so your player picks it up. Container demux and the 16 kHz downmix are pure Rust (symphonia), so there is no ffmpeg to install, and both model hops default to on-device.

> **The public home of `ryu-subtitles`.** Source, builds, and releases live here —
> binaries for every platform are attached to each release.
>
> This tree is generated from the Ryu monorepo, so commits pushed here
> directly are replaced on the next sync. **Pull requests are welcome** —
> open them here and they are ported into the monorepo, then flow back out.
> Ryu as a whole: https://github.com/amajorai/ryu

## Install

- Binary: `ryu-subtitles` from the [Ryu releases](https://github.com/amajorai/ryu/releases).
- Crate: `cargo install ryu-subtitles`.

## License

Apache-2.0 — see [LICENSE](./LICENSE).

---

# Subtitles

Pick a video on this machine, transcribe it, translate the transcript into the
language you want, and get a timed `.srt` / `.vtt` written beside the file — so VLC,
Plex, Jellyfin or Infuse pick the track up with nothing else to do.

Both model hops are local by default. Transcription runs through the extracted
`ryu-stt` crate to local whisper.cpp; translation goes to the node's local gateway
with the bundled on-device model. A node with no provider keys still subtitles a film,
and a video you have not shared with anyone stays that way.

## How it is put together

```
apps-store/subtitles/
  manifest.json        the app: companion runnable + the ryu-subtitles sidecar (port 8013)
  backend/             the sidecar — one process, no lib, ZERO dependency on apps/core
    src/media.rs       container demux → 16 kHz mono → silence-aware windows (symphonia)
    src/cues.rs        segments → subtitle cues; the SubRip / WebVTT writers
    src/translate.rs   the local-gateway translation pass
    src/engine.rs      the job worker: decode → transcribe → translate → write
    src/store.rs       SQLite jobs + node settings
    src/library.rs     the file picker's browse surface and the path gate
    src/api.rs         /api/subtitles/*
  ui/                  the companion (Vite + React, built to ONE self-contained HTML)
```

Core reaches the sidecar through the generic ext-proxy `public_mount`
(`/api/subtitles/*`); the companion reaches it through the single `subtitles.request`
bridge forwarder, because its frame runs under `connect-src 'none'` and has no network
of its own.

## Four decisions worth knowing about

**Whisper, not parakeet.** A subtitle file *is* its timings. `ryu-stt` is built at
default features so `default_stt_engine()` resolves to whisper, which returns
per-segment `start_ms`/`end_ms` from `verbose_json`. Parakeet returns text only — its
`segments` is always empty — so a parakeet job degrades to one coarse cue per window
rather than real subtitles.

**Symphonia, not ffmpeg.** There is no ffmpeg on a normal install: the only one in the
tree is `apps/shadow`'s optional `video` feature, which needs system libraries a
shipped sidecar cannot assume, and `ryu-meetings` reads WAV alone. Symphonia is pure
Rust, so the sidecar stays a single self-contained binary. A container or codec outside
its compiled feature set is reported as *"this file's container or audio codec is not
supported"* — a clear refusal, never a hang or a garbage transcript.

**The job carries a path, not the file.** The sidecar is on the same machine as the
video, so it opens it directly. Nothing is uploaded, nothing is copied, and a 4 GB film
never crosses the companion's sandbox boundary. The path is canonicalized *first* (so
`..` and symlinks resolve), then checked for containment under the user's home
directory or the node data dir, then checked against the media extension list.

**The model never sees a timecode.** Cues go out to the translator as a numbered list
of text and come back as a numbered list of text; the timings stay in Rust. Replies are
matched by index, not position, so a model that merges or splits a line loses that one
line rather than shifting every cue after it. A line that still fails keeps its original
wording, and the job reports `translated_count` below `cue_count` rather than claiming
a clean run.

## Working on it

```bash
cargo test -p ryu-subtitles              # the sidecar
bun test --cwd apps-store/subtitles/ui   # the companion's client layer
bun run --cwd apps-store/subtitles/ui build
scripts/sync-app-fixtures.sh subtitles   # rebuild + refresh the compiled-in bundle
```

Run the sidecar standalone with `RYU_SUBTITLES_PORT` (default 8013), `RYU_DIR`,
`RYU_WHISPER_URL`, `RYU_GATEWAY_URL` / `RYU_GATEWAY_TOKEN` and `RYU_SUBTITLES_MODEL`.
Note that every `/api/subtitles/*` route is **fail-closed** without `RYU_EXT_TOKEN`:
Core injects that token when it spawns the sidecar, and without it the process serves
`/health` and nothing else — deliberately, because this process opens files off the
disk by path.
