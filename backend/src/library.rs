//! Picking the video: the browsable roots, the directory listing, and the path gate.
//!
//! # Why the sidecar browses instead of the companion uploading
//!
//! The obvious shape — an `<input type=file>` in the companion, POSTed to the
//! sidecar — does not survive contact with the file sizes involved. The companion
//! frame runs under `connect-src 'none'`, so its bytes travel through the host
//! bridge, and the one existing host verb that opens a file dialog
//! (`ui.uploadFile`) copies the file into the Uploads space and hands back a
//! `data:` URL. For a 4 GB film that is a base64 string in a renderer process.
//!
//! The sidecar runs on the SAME machine as the video, so it can simply open it. The
//! companion therefore browses — the picker below is the file dialog — and a job
//! carries a PATH. Nothing is copied, nothing is uploaded, and a two-hour film costs
//! one `File::open`.
//!
//! # The path gate
//!
//! Accepting a path from a request means the gate is the security boundary, so it is
//! a canonicalize-then-contain check, not a string check:
//!
//! - the path is canonicalized FIRST, which resolves `..` and symlinks — a
//!   containment test run before canonicalization is defeated by
//!   `~/Movies/../../etc/shadow`, and one run against a symlink target is defeated
//!   by a link planted in a browsable directory;
//! - the result must sit under one of the [`roots`] (the user's own media
//!   directories and the node's data dir), so a request cannot enumerate the
//!   filesystem;
//! - and its extension must be in [`crate::media::MEDIA_EXTENSIONS`], so the decoder
//!   is never pointed at something that is not media.
//!
//! All three, in that order. The sidecar is already loopback-bound and
//! bearer-gated, so this is defence in depth rather than the only lock — but "the
//! transcriber will read any file on the disk and put its contents in a subtitle
//! file" is exactly the shape of bug that turns a local convenience into an
//! exfiltration primitive.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::media::is_media_path;

/// One row in the picker: either a directory to descend into, or a media file to
/// subtitle.
#[derive(Debug, Clone, Serialize)]
pub struct Entry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    /// Bytes, for files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Epoch-millis mtime, so the picker can sort by "what I just downloaded".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_ms: Option<i64>,
}

/// A browsable starting point, with the label the picker shows.
#[derive(Debug, Clone, Serialize)]
pub struct Root {
    pub name: String,
    pub path: String,
}

/// Why a path was refused. Mapped to 400/403/404 at the API edge so the companion
/// can say something true rather than "request failed".
#[derive(Debug, PartialEq, Eq)]
pub enum PathError {
    /// Outside every browsable root.
    OutsideRoots,
    /// Does not exist, or is not readable.
    Missing,
    /// Exists, but is not a file this app will decode.
    NotMedia,
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutsideRoots => write!(
                f,
                "that path is outside the folders this app can read (your home folder and the Ryu data folder)"
            ),
            Self::Missing => write!(f, "that file no longer exists"),
            Self::NotMedia => write!(f, "that file is not a video or audio file this app can read"),
        }
    }
}

/// The directories the picker offers, in the order it shows them. Only those that
/// actually exist: a Linux box has no `~/Movies`, and offering a dead row that
/// errors on click is worse than not offering it.
#[must_use]
pub fn roots() -> Vec<Root> {
    let mut out = Vec::new();
    let mut add = |name: &str, path: Option<PathBuf>| {
        if let Some(path) = path {
            if path.is_dir() {
                out.push(Root {
                    name: name.to_string(),
                    path: path.to_string_lossy().into_owned(),
                });
            }
        }
    };
    add("Movies", dirs::home_dir().map(|h| h.join("Movies")));
    add("Videos", dirs::home_dir().map(|h| h.join("Videos")));
    add("Downloads", dirs::download_dir());
    add("Desktop", dirs::desktop_dir());
    add("Documents", dirs::document_dir());
    // Clips this node recorded itself — the case where the video the user wants
    // subtitled was produced by Ryu in the first place.
    add("Ryu clips", Some(crate::paths::ryu_dir().join("clips")));
    add("Home", dirs::home_dir());
    out
}

/// The containment set for [`validate_source`]: the user's home directory and the
/// node's data dir. Broader than [`roots`] on purpose — a user whose media lives in
/// `~/Media/Films` should be able to paste that path — but still bounded.
fn containment_roots() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(home) = dirs::home_dir() {
        out.push(home);
    }
    out.push(crate::paths::ryu_dir());
    // External drives are where films actually live on macOS; `/Volumes/<disk>` is
    // outside home. Included as a root but NOT as a picker entry, so it must be
    // typed/pasted deliberately.
    #[cfg(target_os = "macos")]
    out.push(PathBuf::from("/Volumes"));
    #[cfg(target_os = "linux")]
    {
        out.push(PathBuf::from("/media"));
        out.push(PathBuf::from("/mnt"));
    }
    out
}

/// List `dir`: subdirectories and media files, directories first, each side sorted
/// by name. Hidden entries are skipped — a picker that shows `.DS_Store` and
/// `.cache` is a worse picker.
///
/// The directory itself goes through the same containment check as a source file, so
/// browsing cannot be used to enumerate `/etc` either.
pub fn list_dir(dir: &Path) -> Result<Vec<Entry>, PathError> {
    let canonical = dir.canonicalize().map_err(|_| PathError::Missing)?;
    if !contained(&canonical) {
        return Err(PathError::OutsideRoots);
    }
    if !canonical.is_dir() {
        return Err(PathError::Missing);
    }

    let read = std::fs::read_dir(&canonical).map_err(|_| PathError::Missing)?;
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for entry in read.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let modified_ms = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64);
        if meta.is_dir() {
            dirs.push(Entry {
                name,
                path: path.to_string_lossy().into_owned(),
                is_dir: true,
                size: None,
                modified_ms,
            });
        } else if is_media_path(&path) {
            files.push(Entry {
                name,
                path: path.to_string_lossy().into_owned(),
                is_dir: false,
                size: Some(meta.len()),
                modified_ms,
            });
        }
    }
    dirs.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    files.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    dirs.extend(files);
    Ok(dirs)
}

/// Canonicalize and check a path a job wants to transcribe. Returns the canonical
/// path, which is what gets stored — so a job records the file it actually read, not
/// the string that pointed at it.
pub fn validate_source(path: &str) -> Result<PathBuf, PathError> {
    let raw = PathBuf::from(path.trim());
    if raw.as_os_str().is_empty() {
        return Err(PathError::Missing);
    }
    // Canonicalize FIRST. Resolving `..` and symlinks before the containment test is
    // the whole gate: the reverse order is trivially escapable.
    let canonical = raw.canonicalize().map_err(|_| PathError::Missing)?;
    if !canonical.is_file() {
        return Err(PathError::Missing);
    }
    if !contained(&canonical) {
        return Err(PathError::OutsideRoots);
    }
    if !is_media_path(&canonical) {
        return Err(PathError::NotMedia);
    }
    Ok(canonical)
}

/// Whether `candidate` (already canonical) sits under a containment root. Roots are
/// canonicalized too — on macOS `dirs::home_dir()` is `/Users/x` while a canonical
/// path under a symlinked home may be `/System/Volumes/Data/Users/x`, and comparing
/// the two raw would reject every legitimate file.
fn contained(candidate: &Path) -> bool {
    containment_roots().iter().any(|root| {
        let root = root.canonicalize().unwrap_or_else(|_| root.clone());
        candidate.starts_with(&root)
    })
}

/// Where a finished subtitle file is written beside its source: the video's own
/// path with the language code and the format's extension appended
/// (`Film.mkv` → `Film.es.srt`).
///
/// That naming is not decorative — it is the convention VLC, Plex, Jellyfin and
/// Infuse use to auto-load a subtitle track and label it with the right language.
#[must_use]
pub fn sidecar_output_path(source: &Path, language_code: &str, extension: &str) -> PathBuf {
    let stem = source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "subtitles".to_string());
    let code = language_code.trim();
    let name = if code.is_empty() {
        format!("{stem}.{extension}")
    } else {
        format!("{stem}.{code}.{extension}")
    };
    source.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory under the OS temp dir, unique per test.
    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ryu-subtitles-lib-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn output_path_follows_the_player_convention() {
        let out = sidecar_output_path(Path::new("/films/The Film.mkv"), "es", "srt");
        assert_eq!(out, PathBuf::from("/films/The Film.es.srt"));
        let vtt = sidecar_output_path(Path::new("/films/The Film.mkv"), "zh-Hans", "vtt");
        assert_eq!(vtt, PathBuf::from("/films/The Film.zh-Hans.vtt"));
    }

    #[test]
    fn output_path_survives_a_missing_language_code() {
        let out = sidecar_output_path(Path::new("/films/clip.mp4"), "  ", "srt");
        assert_eq!(out, PathBuf::from("/films/clip.srt"));
    }

    /// An existing FILE outside every containment root, per platform.
    ///
    /// These three tests hardcoded `/etc/hosts`, which does not exist on Windows —
    /// so the gate returned [`PathError::Missing`] rather than the containment
    /// refusal under test. Still a refusal, so the tests looked fine on a Unix
    /// laptop while proving nothing on Windows, where they failed outright.
    /// The path must EXIST, or the gate short-circuits on `Missing` before it ever
    /// reaches the containment check.
    fn outside_root_file() -> &'static str {
        if cfg!(windows) {
            r"C:\Windows\System32\drivers\etc\hosts"
        } else {
            "/etc/hosts"
        }
    }

    /// An existing DIRECTORY outside every containment root, same reasoning.
    fn outside_root_dir() -> &'static str {
        if cfg!(windows) {
            r"C:\Windows"
        } else {
            "/etc"
        }
    }

    #[test]
    fn a_path_outside_every_root_is_refused() {
        let err = validate_source(outside_root_file()).expect_err("must refuse");
        // Refused either for containment or for not being media — both are a
        // refusal, and which one fires first is not the contract.
        assert!(matches!(err, PathError::OutsideRoots | PathError::NotMedia));
    }

    #[test]
    fn traversal_out_of_a_root_is_refused_after_canonicalization() {
        let home = dirs::home_dir().expect("home");
        // Re-descend onto the real outside file by its root-relative components, so
        // the `..` chain lands somewhere that exists on this platform. Both POSIX
        // and Windows clamp `..` at the root (`/` and the drive respectively), so
        // climbing further than necessary is safe.
        let relative: PathBuf = Path::new(outside_root_file())
            .components()
            .filter(|c| matches!(c, std::path::Component::Normal(_)))
            .collect();
        let escape = format!("{}/../../../../{}", home.display(), relative.display());
        let err = validate_source(&escape).expect_err("must refuse");
        assert!(matches!(err, PathError::OutsideRoots | PathError::NotMedia));
    }

    #[test]
    fn a_non_media_file_inside_a_root_is_refused() {
        let dir = scratch("notmedia");
        let path = dir.join("notes.txt");
        std::fs::write(&path, b"hello").expect("write");
        let result = validate_source(&path.to_string_lossy());
        std::fs::remove_file(&path).ok();
        // The temp dir may or may not sit under a containment root depending on the
        // platform; either refusal is correct, an ACCEPT is not.
        assert!(result.is_err());
    }

    #[test]
    fn a_missing_path_is_missing_not_a_panic() {
        assert_eq!(
            validate_source("/nope/never/at/all.mp4"),
            Err(PathError::Missing)
        );
        assert_eq!(validate_source("   "), Err(PathError::Missing));
    }

    #[test]
    fn a_media_file_under_home_validates_and_comes_back_canonical() {
        let Some(home) = dirs::home_dir() else {
            return;
        };
        let dir = home.join(".ryu-subtitles-test");
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("clip.wav");
        std::fs::write(&path, b"not really a wav, but the gate is extension-based").expect("write");

        let validated = validate_source(&path.to_string_lossy()).expect("should accept");
        assert!(validated.is_absolute());
        assert!(validated.ends_with("clip.wav"));

        // The same file reached through a `..` detour must resolve to the same
        // canonical path — the gate normalizes rather than rejecting.
        let detour = format!("{}/../.ryu-subtitles-test/clip.wav", dir.display());
        assert_eq!(validate_source(&detour).expect("accept detour"), validated);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn listing_hides_dotfiles_and_non_media_and_puts_directories_first() {
        let Some(home) = dirs::home_dir() else {
            return;
        };
        let dir = home.join(".ryu-subtitles-list-test");
        std::fs::create_dir_all(dir.join("zzz-folder")).expect("subdir");
        std::fs::write(dir.join("a.mp4"), b"x").expect("media");
        std::fs::write(dir.join("b.txt"), b"x").expect("text");
        std::fs::write(dir.join(".hidden.mp4"), b"x").expect("hidden");

        let entries = list_dir(&dir).expect("list");
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["zzz-folder", "a.mp4"], "got {names:?}");
        assert!(entries[0].is_dir);
        assert_eq!(entries[1].size, Some(1));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn listing_a_directory_outside_the_roots_is_refused() {
        let err = list_dir(Path::new(outside_root_dir())).expect_err("must refuse");
        assert_eq!(err, PathError::OutsideRoots);
    }

    #[test]
    fn roots_are_all_real_directories() {
        for root in roots() {
            assert!(
                Path::new(&root.path).is_dir(),
                "{} is offered but does not exist",
                root.path
            );
        }
    }
}
