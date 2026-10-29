//! Shared Ryu sidecar data-directory seam.

use std::path::PathBuf;

pub fn ryu_dir() -> PathBuf {
    ryu_sidecar_runtime::ryu_dir()
}

/// Shadow's recorded clips: `<data dir>/services/shadow/media/clips`.
pub fn shadow_clips_dir() -> PathBuf {
    ryu_sidecar_runtime::layout::service_dir(&ryu_dir(), "shadow")
        .join("media")
        .join("clips")
}

/// This app's own data dir: `<data dir>/plugin-data/@ryu+subtitles` (`RYU_PLUGIN_DATA_DIR`
/// when Core spawned the process). Every file the app persists lives under it.
pub fn data_dir() -> PathBuf {
    ryu_sidecar_runtime::plugin_data_dir("@ryu/subtitles")
}
