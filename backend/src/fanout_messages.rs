//! Inbox, toast and push copy as localizable messages.
//!
//! Core renders a notify hint's `i18n` text per recipient (the inbox row and
//! toast in the node's `ui-language`, each push in the locale its device
//! registered with); the English rendering still goes out as the plain
//! `title`/`body`, which an older Core delivers as is.
//!
//! Each `*_MESSAGE` is `format!` with one named argument per placeholder
//! (`name = arg("name")`) so the i18n extractor catalogs the ICU source;
//! `tools/generate-core-i18n.mjs` copies the translations into Core's catalog.
//! Every message is a whole sentence. Core's renderer has no plural selector,
//! so counts are written as `label: {count}`.
#![allow(clippy::useless_format)]

use ryu_app_events::{NotifyHint, NotifyMessage, NotifyTemplate as Template};

pub(crate) const SUBTITLES_READY_MESSAGE: Template = |_| format!("Subtitles ready");
pub(crate) const CUES_MESSAGE: Template = |arg| {
    format!(
        "Cues in {format}: {count}",
        format = arg("format"),
        count = arg("count")
    )
};
pub(crate) const SUBTITLE_JOB_FAILED_MESSAGE: Template = |_| format!("Subtitle job failed");
pub(crate) const OPEN_SUBTITLES_MESSAGE: Template =
    |_| format!("Open Subtitles to see what went wrong.");

/// A finished job. Names no video, path or transcript (all sealed).
pub(crate) fn job_completed(cue_count: usize, extension: &str) -> NotifyHint {
    NotifyHint::localized(
        NotifyMessage::new(SUBTITLES_READY_MESSAGE),
        vec![NotifyMessage::new(CUES_MESSAGE)
            .text("format", extension.to_uppercase())
            .count("count", i64::try_from(cue_count).unwrap_or(i64::MAX))],
    )
}

/// A failed job. The error text is sealed (it can name the video's path).
pub(crate) fn job_failed() -> NotifyHint {
    NotifyHint::localized(
        NotifyMessage::new(SUBTITLE_JOB_FAILED_MESSAGE),
        vec![NotifyMessage::new(OPEN_SUBTITLES_MESSAGE)],
    )
    .with_level("error")
}
