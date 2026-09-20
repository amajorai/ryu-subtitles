//! The target-language table.
//!
//! A closed list, not free text. Two reasons: the picker needs native names to be
//! usable by the person who wants subtitles in that language, and the translation
//! prompt needs an unambiguous English name — a model asked to translate "into pt"
//! guesses, and guesses European Portuguese about half the time.
//!
//! Codes are BCP-47 where a region genuinely distinguishes the written form
//! (`pt-BR`, `zh-Hans`), and bare ISO-639-1 otherwise. The code is what lands in the
//! output filename (`film.es.srt`), which is the convention players and Plex/Jellyfin
//! use to offer a subtitle track by language.

use serde::Serialize;

/// One offerable subtitle language.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Language {
    /// BCP-47 tag. Also the filename infix.
    pub code: &'static str,
    /// English name — what the translation prompt names.
    pub name: &'static str,
    /// Endonym — what the picker shows.
    pub native: &'static str,
}

/// Every language this app will translate into.
pub const LANGUAGES: &[Language] = &[
    Language {
        code: "en",
        name: "English",
        native: "English",
    },
    Language {
        code: "es",
        name: "Spanish",
        native: "Español",
    },
    Language {
        code: "fr",
        name: "French",
        native: "Français",
    },
    Language {
        code: "de",
        name: "German",
        native: "Deutsch",
    },
    Language {
        code: "it",
        name: "Italian",
        native: "Italiano",
    },
    Language {
        code: "pt-BR",
        name: "Brazilian Portuguese",
        native: "Português (Brasil)",
    },
    Language {
        code: "pt",
        name: "European Portuguese",
        native: "Português (Portugal)",
    },
    Language {
        code: "nl",
        name: "Dutch",
        native: "Nederlands",
    },
    Language {
        code: "pl",
        name: "Polish",
        native: "Polski",
    },
    Language {
        code: "ru",
        name: "Russian",
        native: "Русский",
    },
    Language {
        code: "uk",
        name: "Ukrainian",
        native: "Українська",
    },
    Language {
        code: "tr",
        name: "Turkish",
        native: "Türkçe",
    },
    Language {
        code: "ar",
        name: "Arabic",
        native: "العربية",
    },
    Language {
        code: "he",
        name: "Hebrew",
        native: "עברית",
    },
    Language {
        code: "fa",
        name: "Persian",
        native: "فارسی",
    },
    Language {
        code: "hi",
        name: "Hindi",
        native: "हिन्दी",
    },
    Language {
        code: "bn",
        name: "Bengali",
        native: "বাংলা",
    },
    Language {
        code: "ta",
        name: "Tamil",
        native: "தமிழ்",
    },
    Language {
        code: "th",
        name: "Thai",
        native: "ไทย",
    },
    Language {
        code: "vi",
        name: "Vietnamese",
        native: "Tiếng Việt",
    },
    Language {
        code: "id",
        name: "Indonesian",
        native: "Bahasa Indonesia",
    },
    Language {
        code: "ms",
        name: "Malay",
        native: "Bahasa Melayu",
    },
    Language {
        code: "ja",
        name: "Japanese",
        native: "日本語",
    },
    Language {
        code: "ko",
        name: "Korean",
        native: "한국어",
    },
    Language {
        code: "zh-Hans",
        name: "Simplified Chinese",
        native: "简体中文",
    },
    Language {
        code: "zh-Hant",
        name: "Traditional Chinese",
        native: "繁體中文",
    },
    Language {
        code: "sv",
        name: "Swedish",
        native: "Svenska",
    },
    Language {
        code: "no",
        name: "Norwegian",
        native: "Norsk",
    },
    Language {
        code: "da",
        name: "Danish",
        native: "Dansk",
    },
    Language {
        code: "fi",
        name: "Finnish",
        native: "Suomi",
    },
    Language {
        code: "cs",
        name: "Czech",
        native: "Čeština",
    },
    Language {
        code: "el",
        name: "Greek",
        native: "Ελληνικά",
    },
    Language {
        code: "ro",
        name: "Romanian",
        native: "Română",
    },
    Language {
        code: "hu",
        name: "Hungarian",
        native: "Magyar",
    },
];

/// Look a language up by code, case-insensitively. `None` is a 400 at the API edge:
/// an unknown code must never reach the prompt, where it would become a silent
/// translation into whatever the model felt like.
#[must_use]
pub fn find(code: &str) -> Option<&'static Language> {
    let code = code.trim();
    LANGUAGES.iter().find(|l| l.code.eq_ignore_ascii_case(code))
}

/// The default target when the caller and the node settings both say nothing.
pub const DEFAULT_TARGET: &str = "en";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_is_case_insensitive_and_closed() {
        assert_eq!(find("ES").map(|l| l.name), Some("Spanish"));
        assert_eq!(find("zh-hans").map(|l| l.code), Some("zh-Hans"));
        assert!(find("klingon").is_none());
        assert!(find("").is_none());
    }

    #[test]
    fn the_default_target_resolves() {
        assert!(find(DEFAULT_TARGET).is_some());
    }

    #[test]
    fn codes_are_unique_and_names_disambiguate_regional_variants() {
        let mut codes: Vec<&str> = LANGUAGES.iter().map(|l| l.code).collect();
        codes.sort_unstable();
        let before = codes.len();
        codes.dedup();
        assert_eq!(before, codes.len(), "duplicate language code");

        // The pt/pt-BR pair is the reason `name` exists at all.
        assert_eq!(find("pt-BR").map(|l| l.name), Some("Brazilian Portuguese"));
        assert_eq!(find("pt").map(|l| l.name), Some("European Portuguese"));
    }
}
