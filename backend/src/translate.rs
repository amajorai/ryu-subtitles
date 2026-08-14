//! The translation pass: cue text in the spoken language → cue text in the target
//! language, through the LOCAL gateway.
//!
//! Every model call goes to `POST {gateway}/v1/chat/completions` — the same place
//! `ryu-meetings`' note generation goes, and for the same reason: it is the governed
//! egress path, so budgets, DLP and provider routing attach to a subtitle job the
//! way they attach to everything else. The default model is the bundled LOCAL one,
//! so a user with no provider configured still gets translated subtitles from a
//! machine that never talked to anyone.
//!
//! # Why the model never sees a timecode
//!
//! Cues go out as a numbered list of TEXT and come back as a numbered list of text.
//! The timings stay in Rust. A model handed `00:01:23,456 --> 00:01:25,900` will,
//! some fraction of the time, "helpfully" adjust it — and a subtitle file with
//! plausible wrong timings cannot be detected downstream by anything except a human
//! watching the film.
//!
//! # Why the reply is index-matched, not position-matched
//!
//! The failure mode of a batch translation is the model merging two short lines into
//! one, or splitting one long line into two. Position-matching silently shifts every
//! subsequent cue in the batch — the file stays *valid* and becomes *wrong* halfway
//! through. So each line goes out as `<n>⟩ <text>`, the reply is parsed back by that
//! number, and any cue whose number did not come back is retried ALONE. A cue that
//! still fails keeps its source text (see [`Layout::Translated`]'s fallback in
//! `cues.rs`), which is a visibly untranslated line rather than a wrong one.
//!
//! [`Layout::Translated`]: crate::cues::Layout::Translated

use std::collections::HashMap;

use serde_json::json;

use crate::cues::Cue;
use crate::languages::Language;

/// How many cues go in one request. Small enough that a local 2B model keeps the
/// numbering straight, large enough that a 90-minute film is ~40 calls and not 1200.
const BATCH: usize = 25;

/// Separator between a cue's index and its text. `⟩` rather than `.` or `:` because
/// it never appears in dialogue, so parsing the reply cannot be confused by a line
/// that legitimately starts with a number.
const MARKER: char = '⟩';

/// The local gateway default (mirrors `apps/core/src/sidecar/gateway.rs`).
pub const DEFAULT_GATEWAY_URL: &str = "http://127.0.0.1:7981";

/// The bundled local model, mirroring Core's `registry::DEFAULT_LLM_MODEL`. Nothing
/// here is hardcoded to a remote provider: this is the on-device default, and a node
/// setting or `RYU_SUBTITLES_MODEL` overrides it.
pub const DEFAULT_MODEL: &str = "gemma-4-E2B-it-Q4_K_M";

/// Where and how to reach the gateway for one job.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub url: String,
    pub token: Option<String>,
    pub model: String,
}

impl GatewayConfig {
    /// Resolve from the environment Core injects at spawn, with `model` overriding
    /// the bundled default when the caller (job or node settings) named one.
    #[must_use]
    pub fn from_env(model: Option<String>) -> Self {
        let url = std::env::var("RYU_GATEWAY_URL")
            .ok()
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_GATEWAY_URL.to_string());
        let token = std::env::var("RYU_GATEWAY_TOKEN")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let model = model
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .or_else(|| {
                std::env::var("RYU_SUBTITLES_MODEL")
                    .ok()
                    .map(|m| m.trim().to_string())
                    .filter(|m| !m.is_empty())
            })
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
        Self { url, token, model }
    }
}

/// The system prompt. Explicit about the two things that break a subtitle file:
/// dropping/merging lines, and translating the numbering itself.
fn system_prompt(target: &Language) -> String {
    format!(
        "You translate film and video subtitles into {} ({}).\n\
         You are given numbered subtitle lines, one per line, in the form `N{MARKER} text`.\n\
         Rules:\n\
         - Reply with exactly one line per input line, in the same `N{MARKER} text` form, \
           using the SAME numbers you were given.\n\
         - Never merge two input lines into one, never split one into two, never omit one, \
           never add one.\n\
         - Translate only the text after the marker. Keep proper nouns, numbers and \
           on-screen names as they are.\n\
         - These lines are consecutive dialogue from one video, so a line may be a \
           sentence fragment: translate it as a fragment, do not complete it.\n\
         - Match the register of the original — subtitles are spoken language, not prose.\n\
         - Output nothing but the numbered lines: no preamble, no notes, no code fences.",
        target.name, target.native
    )
}

/// Translate every cue in `cues` in place, filling [`Cue::translated`].
///
/// Best-effort by design: a cue the model never returned keeps `translated: None`
/// and renders as its source text. Returns how many cues were translated, so the
/// job can record "1180 of 1204 lines translated" rather than claiming a clean run.
///
/// `on_progress` is called with `(done, total)` after each batch — a 90-minute film
/// is minutes of model time, and a progress bar that sits at 0 reads as a hang.
pub async fn translate_cues(
    client: &reqwest::Client,
    config: &GatewayConfig,
    target: &Language,
    cues: &mut [Cue],
    mut on_progress: impl FnMut(usize, usize),
) -> usize {
    let total = cues.len();
    let mut translated = 0usize;
    let system = system_prompt(target);

    for (batch_index, batch) in cues.chunks_mut(BATCH).enumerate() {
        let base = batch_index * BATCH;
        let numbered = render_batch(batch, base);
        let mut got = match complete(client, config, &system, &numbered).await {
            Ok(reply) => parse_reply(&reply),
            Err(e) => {
                tracing::warn!(error = %e, "subtitles: translation batch failed; retrying line by line");
                HashMap::new()
            }
        };

        // Retry, alone, every line the batch did not answer for. This is the
        // merge/split repair: a lone line has no numbering for the model to lose.
        for (offset, cue) in batch.iter().enumerate() {
            let n = base + offset + 1;
            if got.contains_key(&n) {
                continue;
            }
            let single = format!("{n}{MARKER} {}", cue.text);
            if let Ok(reply) = complete(client, config, &system, &single).await {
                let single_got = parse_reply(&reply);
                if let Some(text) = single_got.get(&n).cloned() {
                    got.insert(n, text);
                } else if let Some(text) = single_line_fallback(&reply) {
                    // A model that dropped the numbering on a single-line request
                    // still answered the question; the mapping is unambiguous here
                    // precisely because there is only one line.
                    got.insert(n, text);
                }
            }
        }

        for (offset, cue) in batch.iter_mut().enumerate() {
            if let Some(text) = got.get(&(base + offset + 1)) {
                let text = text.trim();
                if !text.is_empty() {
                    cue.translated = Some(text.to_string());
                    translated += 1;
                }
            }
        }
        on_progress((base + batch.len()).min(total), total);
    }
    translated
}

/// `1⟩ line one\n2⟩ line two…`, numbered from `base + 1`.
fn render_batch(cues: &[Cue], base: usize) -> String {
    cues.iter()
        .enumerate()
        .map(|(i, c)| format!("{}{MARKER} {}", base + i + 1, c.text.replace('\n', " ")))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse `N⟩ text` lines back into a number → text map. Lines without the marker are
/// ignored, which is what makes a chatty preamble harmless rather than corrupting.
fn parse_reply(reply: &str) -> HashMap<usize, String> {
    let mut out = HashMap::new();
    for line in reply.lines() {
        let line = line.trim();
        let Some((head, tail)) = line.split_once(MARKER) else {
            continue;
        };
        let Ok(n) = head.trim().trim_start_matches(['-', '*', '•']).trim().parse::<usize>() else {
            continue;
        };
        let text = tail.trim();
        if !text.is_empty() {
            out.insert(n, text.to_string());
        }
    }
    out
}

/// The single-line retry's escape hatch: the first non-empty line of a reply that
/// carries no numbering at all.
fn single_line_fallback(reply: &str) -> Option<String> {
    reply
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("```"))
        .map(std::string::ToString::to_string)
}

/// One chat completion against the local gateway.
async fn complete(
    client: &reqwest::Client,
    config: &GatewayConfig,
    system: &str,
    user: &str,
) -> anyhow::Result<String> {
    let url = format!("{}/v1/chat/completions", config.url.trim_end_matches('/'));
    let mut request = client.post(&url).json(&json!({
        "model": config.model,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user },
        ],
        // Deterministic: two runs over the same film should produce the same file,
        // and creativity is not a virtue in a subtitle.
        "temperature": 0.0,
        "stream": false,
    }));
    // The local gateway accepts the `ryu-local` dev bearer; a configured token wins.
    let bearer = config.token.clone().unwrap_or_else(|| "ryu-local".to_string());
    request = request.bearer_auth(bearer);

    let response = request.send().await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!("gateway {status}: {}", body.chars().take(400).collect::<String>());
    }
    let value: serde_json::Value = serde_json::from_str(&body)?;
    let content = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    if content.trim().is_empty() {
        anyhow::bail!("the gateway returned an empty completion");
    }
    Ok(content)
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
    fn batches_are_numbered_from_their_absolute_position() {
        let cues = vec![cue("one"), cue("two")];
        let rendered = render_batch(&cues, 50);
        assert_eq!(rendered, "51⟩ one\n52⟩ two");
    }

    #[test]
    fn newlines_inside_a_cue_are_flattened_so_one_cue_is_one_line() {
        let rendered = render_batch(&[cue("first\nsecond")], 0);
        assert_eq!(rendered.lines().count(), 1);
        assert_eq!(rendered, "1⟩ first second");
    }

    #[test]
    fn reply_parsing_is_by_index_not_position() {
        // The model answered out of order and skipped 2. Index-matching must place
        // 3 on cue 3 — position-matching would put it on cue 2.
        let got = parse_reply("3⟩ tres\n1⟩ uno\n");
        assert_eq!(got.get(&1).map(String::as_str), Some("uno"));
        assert_eq!(got.get(&3).map(String::as_str), Some("tres"));
        assert!(!got.contains_key(&2));
    }

    #[test]
    fn chatty_preamble_and_code_fences_are_ignored() {
        let got = parse_reply("Sure! Here are the lines:\n```\n1⟩ hola\n```\nLet me know!");
        assert_eq!(got.len(), 1);
        assert_eq!(got.get(&1).map(String::as_str), Some("hola"));
    }

    #[test]
    fn bulleted_numbering_still_parses() {
        let got = parse_reply("- 1⟩ hola\n* 2⟩ adios");
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn empty_translations_are_not_recorded() {
        let got = parse_reply("1⟩   \n2⟩ real");
        assert!(!got.contains_key(&1));
        assert!(got.contains_key(&2));
    }

    #[test]
    fn a_line_that_legitimately_starts_with_a_number_is_not_confused_for_an_index() {
        // "1984 was a good year" as cue TEXT — no marker, so it is not an index.
        let got = parse_reply("1⟩ 1984 was a good year");
        assert_eq!(got.get(&1).map(String::as_str), Some("1984 was a good year"));
    }

    #[test]
    fn single_line_fallback_takes_the_first_real_line() {
        assert_eq!(
            single_line_fallback("```\nhola mundo\n```"),
            Some("hola mundo".to_string())
        );
        assert_eq!(single_line_fallback("   \n\n"), None);
    }

    #[test]
    fn the_prompt_names_the_language_unambiguously() {
        let pt = crate::languages::find("pt-BR").expect("pt-BR");
        let prompt = system_prompt(pt);
        assert!(prompt.contains("Brazilian Portuguese"));
        assert!(prompt.contains("Português (Brasil)"));
    }

    #[test]
    fn gateway_config_prefers_an_explicit_model_over_the_bundled_default() {
        let explicit = GatewayConfig::from_env(Some("qwen3-4b".into()));
        assert_eq!(explicit.model, "qwen3-4b");
        let blank = GatewayConfig::from_env(Some("   ".into()));
        assert!(!blank.model.is_empty(), "blank must fall through, not stick");
    }
}
