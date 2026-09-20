//! The cue model, the layout rules, and the SubRip / WebVTT writers.
//!
//! An STT engine returns transcript segments, not subtitles. The difference is
//! layout: whisper happily emits a single 24-second segment holding three sentences,
//! which as a subtitle is a wall of text that outlives the shot it belongs to.
//! Everything here is the conversion between the two — splitting long segments,
//! enforcing a floor and ceiling on how long a cue stays on screen, wrapping to at
//! most two lines, and dropping the artifacts (empty cues, whisper's `[BLANK_AUDIO]`
//! placeholders) that would otherwise become blank frames in the file.
//!
//! The numbers below are the conventional broadcast values (BBC/Netflix house style
//! agrees within rounding): ~42 characters per line, two lines, at least ~5/6 s and
//! at most 7 s on screen. They are constants rather than settings because a user
//! cannot answer "what should the maximum characters per line be", and the file is
//! wrong in a way they will feel but not diagnose if the answer is bad.
//!
//! ## Timings are never sent to a model
//!
//! The translation pass ([`crate::translate`]) receives cue TEXT and returns cue
//! text. The timings on the [`Cue`]s here are the ones the STT engine measured, and
//! they survive translation untouched. This is deliberate: a model handed timecodes
//! reformats them, silently, on some fraction of calls — and a subtitle file with
//! plausible-but-wrong timings is worse than no subtitle file, because nothing
//! downstream can detect it.

use serde::{Deserialize, Serialize};

/// Maximum characters on one subtitle line before it wraps.
const MAX_LINE_CHARS: usize = 42;
/// Maximum lines in one cue. A third line covers the picture.
const MAX_LINES: usize = 2;
/// Longest a single cue may stay on screen.
const MAX_CUE_MS: u64 = 7_000;
/// Shortest a cue may stay on screen — below this it reads as a flash.
const MIN_CUE_MS: u64 = 800;
/// Gap left between two cues that would otherwise abut, so the change is visible.
const CUE_GAP_MS: u64 = 40;

/// One subtitle: when it appears, when it leaves, what it says.
///
/// `text` is always the SOURCE-language line the engine produced. `translated` is
/// `Some` only after a translation pass ran, which is what lets a bilingual file be
/// rendered without a second transcription.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cue {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translated: Option<String>,
}

impl Cue {
    /// The line(s) this cue contributes to a rendered file, in `layout` order.
    fn rendered(&self, layout: Layout) -> String {
        let source = wrap(&self.text);
        let translated = self.translated.as_deref().map(wrap);
        match (layout, translated) {
            (Layout::Translated, Some(t)) => t,
            (Layout::Translated, None) => source,
            (Layout::Source, _) => source,
            (Layout::Bilingual, Some(t)) => format!("{t}\n{source}"),
            (Layout::Bilingual, None) => source,
        }
    }
}

/// Which text a rendered file carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// Target language only — the default, and what "subtitle this film in Spanish"
    /// means.
    Translated,
    /// What was actually said, untranslated.
    Source,
    /// Target language on top, source underneath. Language learners want this; it is
    /// also the honest option when the translation is not trusted.
    Bilingual,
}

impl Default for Layout {
    fn default() -> Self {
        Self::Translated
    }
}

/// Subtitle file formats this app writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// SubRip. The universal one — every player reads it.
    Srt,
    /// WebVTT. What a `<track>` element in a browser needs.
    Vtt,
}

impl Format {
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
        }
    }

    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "srt" | "subrip" => Some(Self::Srt),
            "vtt" | "webvtt" => Some(Self::Vtt),
            _ => None,
        }
    }

    /// `text/vtt` matters: a browser refuses a `<track>` served as `text/plain`.
    #[must_use]
    pub fn content_type(self) -> &'static str {
        match self {
            Self::Srt => "application/x-subrip",
            Self::Vtt => "text/vtt",
        }
    }
}

impl Default for Format {
    fn default() -> Self {
        Self::Srt
    }
}

/// Turn one window's engine segments into cues on the media's absolute timeline.
///
/// `offset_ms` is the window's start; engine timestamps are window-relative, so this
/// is what makes cue times absolute. Segments the engine emits for silence
/// (whisper's `[BLANK_AUDIO]`, `(silence)`, bare punctuation) are dropped — they
/// would otherwise be visible as an empty cue that blocks the picture.
pub fn cues_from_segments(
    offset_ms: u64,
    segments: &[ryu_stt::TranscriptSegment],
    fallback_text: &str,
    window_duration_ms: u64,
) -> Vec<Cue> {
    if segments.is_empty() {
        // An engine that returned text but no timings (or a window whose segments
        // were all silence markers) still has something to say. One cue spanning the
        // window is a truthful, if coarse, placement — better than dropping the
        // speech entirely.
        let text = clean(fallback_text);
        if text.is_empty() {
            return Vec::new();
        }
        return split_long(&Cue {
            start_ms: offset_ms,
            end_ms: offset_ms + window_duration_ms.max(MIN_CUE_MS),
            text,
            translated: None,
        });
    }

    let mut out = Vec::new();
    for segment in segments {
        let text = clean(&segment.text);
        if text.is_empty() {
            continue;
        }
        let start = offset_ms + segment.start_ms;
        // A zero-length or inverted segment is not fatal; give it the floor rather
        // than emitting a cue that ends before it starts.
        let end = offset_ms + segment.end_ms.max(segment.start_ms + MIN_CUE_MS);
        out.extend(split_long(&Cue {
            start_ms: start,
            end_ms: end,
            text,
            translated: None,
        }));
    }
    out
}

/// Final pass over a whole file's cues: order them, stop them overlapping, and give
/// each one a sane minimum time on screen.
///
/// Runs once at the end rather than per window, because the constraint that matters
/// most (a cue must not overlap the NEXT one) spans the boundary between two windows
/// transcribed in separate engine calls.
#[must_use]
pub fn normalize(mut cues: Vec<Cue>) -> Vec<Cue> {
    cues.retain(|c| !c.text.trim().is_empty());
    cues.sort_by_key(|c| (c.start_ms, c.end_ms));

    for i in 0..cues.len() {
        if cues[i].end_ms < cues[i].start_ms + MIN_CUE_MS {
            cues[i].end_ms = cues[i].start_ms + MIN_CUE_MS;
        }
        if let Some(next_start) = cues.get(i + 1).map(|c| c.start_ms) {
            // Only ever pull an end EARLIER. Pushing the next cue later would
            // cascade the whole file off the audio.
            if cues[i].end_ms + CUE_GAP_MS > next_start {
                cues[i].end_ms = next_start
                    .saturating_sub(CUE_GAP_MS)
                    .max(cues[i].start_ms + 1);
            }
        }
    }
    cues
}

/// Split a cue whose text is too long for [`MAX_LINES`] lines, or which stays on
/// screen past [`MAX_CUE_MS`], into consecutive cues sharing its span
/// proportionally to their text length.
fn split_long(cue: &Cue) -> Vec<Cue> {
    let budget = MAX_LINE_CHARS * MAX_LINES;
    let duration = cue.end_ms.saturating_sub(cue.start_ms);
    if cue.text.chars().count() <= budget && duration <= MAX_CUE_MS {
        return vec![cue.clone()];
    }

    let parts = split_text(&cue.text, budget);
    if parts.len() <= 1 {
        return vec![cue.clone()];
    }
    let total: usize = parts
        .iter()
        .map(|p| p.chars().count())
        .sum::<usize>()
        .max(1);

    let mut out = Vec::with_capacity(parts.len());
    let mut cursor = cue.start_ms;
    for (i, part) in parts.iter().enumerate() {
        let share = duration * part.chars().count() as u64 / total as u64;
        let end = if i + 1 == parts.len() {
            cue.end_ms
        } else {
            (cursor + share).min(cue.end_ms)
        };
        out.push(Cue {
            start_ms: cursor,
            end_ms: end.max(cursor + 1),
            text: part.clone(),
            translated: None,
        });
        cursor = end;
    }
    out
}

/// Break `text` into chunks of at most `budget` characters, preferring a sentence
/// end, then a clause boundary, then a word gap. Never mid-word: a cue that ends in
/// half a word is the one artifact viewers notice immediately.
fn split_text(text: &str, budget: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text.trim();
    while rest.chars().count() > budget {
        let cut = best_break(rest, budget);
        let (head, tail) = rest.split_at(cut);
        let head = head.trim();
        if head.is_empty() {
            break;
        }
        out.push(head.to_string());
        rest = tail.trim_start();
    }
    if !rest.is_empty() {
        out.push(rest.to_string());
    }
    out
}

/// Byte index to split at: the last sentence end within budget, else the last
/// clause break, else the last space, else the budget itself (rounded down to a
/// character boundary so the split cannot panic on multi-byte text).
fn best_break(text: &str, budget: usize) -> usize {
    let limit = text
        .char_indices()
        .nth(budget)
        .map_or(text.len(), |(i, _)| i);
    let head = &text[..limit];
    for pattern in [". ", "? ", "! ", "。", "？", "！"] {
        if let Some(idx) = head.rfind(pattern) {
            return idx + pattern.len();
        }
    }
    for pattern in [", ", "; ", ": ", "、", "，"] {
        if let Some(idx) = head.rfind(pattern) {
            return idx + pattern.len();
        }
    }
    if let Some(idx) = head.rfind(' ') {
        return idx + 1;
    }
    limit
}

/// Wrap one cue's text to at most [`MAX_LINES`] lines of [`MAX_LINE_CHARS`].
fn wrap(text: &str) -> String {
    let text = text.trim();
    if text.chars().count() <= MAX_LINE_CHARS {
        return text.to_string();
    }
    let mut lines = split_text(text, MAX_LINE_CHARS);
    // `split_long` bounds the cue to two lines' worth of characters before this
    // runs, so overflow here means unbreakable text (a URL, a CJK run with no
    // spaces). Joining the tail onto the last line beats silently deleting words.
    if lines.len() > MAX_LINES {
        let tail = lines.split_off(MAX_LINES - 1).join(" ");
        lines.push(tail);
    }
    lines.join("\n")
}

/// Strip the placeholders engines emit for non-speech, and collapse whitespace.
fn clean(text: &str) -> String {
    let trimmed = text.trim();
    let lowered = trimmed.to_ascii_lowercase();
    const NOISE: &[&str] = &[
        "[blank_audio]",
        "[silence]",
        "(silence)",
        "[music]",
        "(music)",
        "[applause]",
        "[inaudible]",
        "[ silence ]",
        "*",
        ".",
        "..",
        "...",
        "…",
        "-",
    ];
    if NOISE.contains(&lowered.as_str()) {
        return String::new();
    }
    trimmed.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Render `cues` as a subtitle file.
#[must_use]
pub fn render(cues: &[Cue], format: Format, layout: Layout) -> String {
    match format {
        Format::Srt => render_srt(cues, layout),
        Format::Vtt => render_vtt(cues, layout),
    }
}

fn render_srt(cues: &[Cue], layout: Layout) -> String {
    let mut out = String::new();
    let mut index = 1;
    for cue in cues {
        let body = cue.rendered(layout);
        if body.trim().is_empty() {
            continue;
        }
        out.push_str(&format!(
            "{index}\n{} --> {}\n{body}\n\n",
            timestamp(cue.start_ms, ','),
            timestamp(cue.end_ms, ',')
        ));
        index += 1;
    }
    out
}

fn render_vtt(cues: &[Cue], layout: Layout) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for cue in cues {
        let body = cue.rendered(layout);
        if body.trim().is_empty() {
            continue;
        }
        out.push_str(&format!(
            "{} --> {}\n{body}\n\n",
            timestamp(cue.start_ms, '.'),
            timestamp(cue.end_ms, '.')
        ));
    }
    out
}

/// `HH:MM:SS<sep>mmm`. SubRip uses a comma before the milliseconds, WebVTT a dot,
/// and a player given the wrong one shows no subtitles at all rather than an error.
fn timestamp(ms: u64, sep: char) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1000;
    let millis = ms % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{sep}{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start_ms: u64, end_ms: u64, text: &str) -> ryu_stt::TranscriptSegment {
        ryu_stt::TranscriptSegment {
            start_ms,
            end_ms,
            text: text.to_string(),
        }
    }

    fn cue(start_ms: u64, end_ms: u64, text: &str) -> Cue {
        Cue {
            start_ms,
            end_ms,
            text: text.to_string(),
            translated: None,
        }
    }

    #[test]
    fn segment_times_are_offset_onto_the_absolute_timeline() {
        let cues = cues_from_segments(60_000, &[seg(1_000, 3_000, "hello there")], "", 30_000);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].start_ms, 61_000);
        assert_eq!(cues[0].end_ms, 63_000);
    }

    #[test]
    fn silence_markers_never_become_cues() {
        let cues = cues_from_segments(
            0,
            &[
                seg(0, 1_000, "[BLANK_AUDIO]"),
                seg(1_000, 2_000, "real speech"),
            ],
            "",
            30_000,
        );
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "real speech");
    }

    #[test]
    fn a_window_with_text_but_no_segments_still_produces_a_cue() {
        // The parakeet-shaped case: text, no timings. One coarse cue beats losing
        // the speech.
        let cues = cues_from_segments(5_000, &[], "something was said", 30_000);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].start_ms, 5_000);
        assert_eq!(cues[0].end_ms, 35_000);
    }

    #[test]
    fn an_empty_window_produces_nothing() {
        assert!(cues_from_segments(0, &[], "   ", 30_000).is_empty());
        assert!(cues_from_segments(0, &[], "[BLANK_AUDIO]", 30_000).is_empty());
    }

    #[test]
    fn a_long_segment_splits_at_sentence_ends_and_shares_its_span() {
        let long = "This is the first sentence and it runs on for a while. \
                    This is the second sentence which also runs on for a while. \
                    And here is a third one to be sure we exceed the budget.";
        let cues = cues_from_segments(0, &[seg(0, 12_000, long)], "", 30_000);
        assert!(cues.len() >= 2, "expected a split, got {}", cues.len());
        assert_eq!(cues[0].start_ms, 0);
        assert_eq!(cues.last().expect("last").end_ms, 12_000);
        for pair in cues.windows(2) {
            assert!(
                pair[0].end_ms <= pair[1].start_ms,
                "split cues must not overlap"
            );
        }
        for c in &cues {
            assert!(
                !c.text.ends_with(' ') && !c.text.starts_with(' '),
                "split text must be trimmed"
            );
        }
    }

    #[test]
    fn normalize_removes_overlap_by_pulling_the_earlier_cue_in() {
        let cues = normalize(vec![cue(0, 5_000, "one"), cue(3_000, 6_000, "two")]);
        assert!(cues[0].end_ms <= cues[1].start_ms);
        assert_eq!(cues[1].start_ms, 3_000, "the later cue must not move");
    }

    #[test]
    fn normalize_gives_a_flash_cue_the_minimum_screen_time() {
        let cues = normalize(vec![cue(1_000, 1_050, "hi")]);
        assert_eq!(cues[0].end_ms - cues[0].start_ms, MIN_CUE_MS);
    }

    #[test]
    fn normalize_sorts_out_of_order_windows() {
        let cues = normalize(vec![cue(10_000, 12_000, "later"), cue(0, 2_000, "earlier")]);
        assert_eq!(cues[0].text, "earlier");
    }

    #[test]
    fn srt_is_indexed_from_one_with_comma_milliseconds() {
        let out = render(
            &[cue(0, 1_500, "first"), cue(2_000, 3_000, "second")],
            Format::Srt,
            Layout::Source,
        );
        assert!(out.starts_with("1\n00:00:00,000 --> 00:00:01,500\nfirst\n\n"));
        assert!(out.contains("2\n00:00:02,000 --> 00:00:03,000\nsecond"));
    }

    #[test]
    fn vtt_has_the_header_and_dot_milliseconds() {
        let out = render(&[cue(0, 1_500, "first")], Format::Vtt, Layout::Source);
        assert!(out.starts_with("WEBVTT\n\n"));
        assert!(out.contains("00:00:00.000 --> 00:00:01.500"));
        assert!(!out.contains(",500"));
    }

    #[test]
    fn hours_roll_over_correctly() {
        assert_eq!(timestamp(3_723_456, ','), "01:02:03,456");
    }

    #[test]
    fn translated_layout_renders_the_translation_and_bilingual_renders_both() {
        let c = Cue {
            start_ms: 0,
            end_ms: 1_000,
            text: "hello".into(),
            translated: Some("hola".into()),
        };
        assert!(render(&[c.clone()], Format::Srt, Layout::Translated).contains("hola"));
        assert!(!render(&[c.clone()], Format::Srt, Layout::Translated).contains("hello"));
        let both = render(&[c], Format::Srt, Layout::Bilingual);
        assert!(both.contains("hola\nhello"), "target above source: {both}");
    }

    #[test]
    fn an_untranslated_cue_falls_back_to_its_source_text() {
        // What a failed/skipped translation pass must produce: a usable file, not
        // an empty one.
        let out = render(&[cue(0, 1_000, "hello")], Format::Srt, Layout::Translated);
        assert!(out.contains("hello"));
    }

    #[test]
    fn a_long_segment_renders_as_short_cues_of_at_most_two_lines() {
        // Through the real path (segment → split → render), which is what bounds a
        // cue's text to two lines' worth before wrapping ever sees it.
        let text = "word ".repeat(60);
        let cues = normalize(cues_from_segments(
            0,
            &[seg(0, 20_000, text.trim())],
            "",
            30_000,
        ));
        assert!(cues.len() > 1, "a 300-character segment must split");
        let out = render(&cues, Format::Srt, Layout::Source);
        for block in out.split("\n\n").filter(|b| !b.trim().is_empty()) {
            let body: Vec<&str> = block.lines().skip(2).collect();
            assert!(
                body.len() <= MAX_LINES,
                "{} lines in one cue: {block}",
                body.len()
            );
            for line in body {
                assert!(
                    line.chars().count() <= MAX_LINE_CHARS,
                    "line too long ({}): {line}",
                    line.chars().count()
                );
            }
        }
    }

    #[test]
    fn unbreakable_text_keeps_two_lines_rather_than_losing_words() {
        // A cue built directly (no split pass) whose text cannot fit: wrapping must
        // overflow the LAST line rather than silently drop the tail, because a
        // dropped word is invisible and a long line is merely ugly.
        let text = "word ".repeat(30);
        let out = render(&[cue(0, 3_000, text.trim())], Format::Srt, Layout::Source);
        let body: Vec<&str> = out.lines().skip(2).take_while(|l| !l.is_empty()).collect();
        assert!(body.len() <= MAX_LINES, "got {} lines", body.len());
        assert_eq!(
            body.join(" ").split_whitespace().count(),
            30,
            "every word must survive the wrap"
        );
    }

    #[test]
    fn multibyte_text_never_splits_mid_character() {
        // A CJK run has no spaces, so `best_break` falls through to the budget —
        // which must land on a char boundary or the slice panics.
        let text = "日本語のテキストがとても長い場合".repeat(6);
        let parts = split_text(&text, 42);
        assert!(parts.len() > 1);
        assert_eq!(parts.concat().chars().count(), text.chars().count());
    }

    #[test]
    fn format_parse_accepts_both_spellings_and_rejects_junk() {
        assert_eq!(Format::parse("SRT"), Some(Format::Srt));
        assert_eq!(Format::parse("webvtt"), Some(Format::Vtt));
        assert_eq!(Format::parse("ass"), None);
    }
}
