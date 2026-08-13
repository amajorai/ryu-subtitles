//! Container demux → 16 kHz mono PCM → silence-aware windows.
//!
//! This is the half of the app that turns "a video the user picked" into
//! something an STT engine will accept. Three constraints shape it:
//!
//! 1. **There is no ffmpeg.** The only FFmpeg in the tree is `apps/shadow`'s
//!    OPTIONAL `video` feature, which links system libraries a shipped sidecar
//!    cannot assume, and `ryu-meetings` reads WAV alone (`hound`) — it rejects
//!    mp3/m4a/mov outright. So the decode is [`symphonia`]: pure Rust, one binary,
//!    no runtime dependency. A container or codec outside its feature set yields
//!    [`MediaError::Unsupported`], which the API surfaces as 415 — a clear "this
//!    file's audio codec is not supported" beats a hang or a garbage transcript.
//!
//! 2. **Whisper wants 16 kHz mono.** Video audio is 44.1/48 kHz stereo (or 5.1), so
//!    every decoded packet is downmixed to mono and resampled to
//!    [`TARGET_RATE`]. The resampler is a **box average** over the source span each
//!    output sample covers, not a nearest-neighbour pick: dropping 2 of every 3
//!    samples at 48 kHz aliases high frequencies down into the speech band, which
//!    the transcript then reports as hallucinated words.
//!
//! 3. **A 2-hour film cannot live in memory.** 2 h of 16 kHz mono i16 is ~230 MB
//!    *after* downmix, and several times that before it. So decoding STREAMS: a
//!    blocking decode task emits [`Window`]s into a bounded channel, and the job
//!    worker transcribes each as it arrives. Memory is O(one window), and the
//!    bounded channel is what applies backpressure — without it a fast decoder
//!    would race ahead of a slow local whisper and rebuild the very buffer this
//!    design exists to avoid.
//!
//! ## Why windows are cut at silence
//!
//! Whisper is fed fixed windows, and a cut through the middle of a word is
//! transcribed as two wrong words — one at the end of window N and one at the start
//! of N+1, neither recoverable later. So the cut point is not exactly
//! [`WINDOW_SECS`]: it is the quietest 100 ms frame within ±[`CUT_SEARCH_SECS`] of
//! it. Speech has gaps; this finds one. The window still has a hard maximum
//! (`WINDOW_SECS + CUT_SEARCH_SECS`) so continuous audio (music, noise) cannot
//! stretch a window unboundedly.

use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// The sample rate every window is resampled to. Whisper's own front end runs at
/// 16 kHz; handing it anything else means the engine resamples, or worse, does not.
pub const TARGET_RATE: u32 = 16_000;

/// Nominal window length. 30 s matches whisper's native receptive field, so a
/// window is one engine pass rather than an internal re-split whose segment
/// timestamps we would then have to trust across a boundary we did not choose.
pub const WINDOW_SECS: u32 = 30;

/// How far on either side of the nominal cut point to hunt for silence.
const CUT_SEARCH_SECS: u32 = 5;

/// Granularity of the silence hunt. 100 ms is shorter than the gap between spoken
/// words and long enough that one glottal stop does not read as a pause.
const SILENCE_FRAME_MS: u32 = 100;

/// A decoded, resampled window of audio: where it starts in the source, and the
/// 16 kHz mono WAV bytes to hand the STT engine.
#[derive(Debug, Clone)]
pub struct Window {
    /// Offset of this window's first sample from the start of the media, in ms.
    /// Every segment timestamp the engine returns is relative to the window, so
    /// this is what makes a cue's absolute time absolute.
    pub offset_ms: u64,
    /// Duration of the window in ms (its own length, not the source's).
    pub duration_ms: u64,
    /// 16 kHz mono PCM WAV, header included.
    pub wav: Vec<u8>,
}

/// What can go wrong turning a file into windows. Split from a bare `anyhow` so the
/// API can map "we cannot read this kind of file" (415) apart from "the file is
/// broken" (422) and "the disk failed" (500) — a user who picked an AC-3 MKV needs
/// to be told to pick something else, not shown a server error.
#[derive(Debug)]
pub enum MediaError {
    /// The path does not exist, or could not be opened.
    Io(String),
    /// The container or codec is outside symphonia's compiled feature set.
    Unsupported(String),
    /// The file opened but carries no audio track (a silent screen recording, or a
    /// video whose only track is video).
    NoAudioTrack,
    /// The stream decoded partway and then failed.
    Decode(String),
}

impl std::fmt::Display for MediaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "could not open the media file: {e}"),
            Self::Unsupported(e) => write!(
                f,
                "this file's container or audio codec is not supported: {e}"
            ),
            Self::NoAudioTrack => write!(f, "this file has no audio track to transcribe"),
            Self::Decode(e) => write!(f, "the audio stream could not be decoded: {e}"),
        }
    }
}

impl std::error::Error for MediaError {}

/// Container/extension allowlist for the picker and for job creation.
///
/// Extension-based, and that is deliberate: symphonia probes by content, so this
/// list is not what decides whether a decode succeeds. It is what keeps the library
/// browser from listing a user's entire home directory, and what gives job creation
/// a cheap, honest rejection before it spawns a decode task.
pub const MEDIA_EXTENSIONS: &[&str] = &[
    // video containers
    "mp4", "m4v", "mov", "mkv", "webm", "avi", // audio containers
    "m4a", "mp3", "wav", "flac", "aac", "ogg", "oga", "opus", "caf", "aiff",
];

/// Whether `path`'s extension is one this app will attempt.
#[must_use]
pub fn is_media_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| MEDIA_EXTENSIONS.contains(&e.as_str()))
}

/// Decode `path` and emit 16 kHz mono windows through `on_window`, in order.
///
/// Blocking and CPU-bound — call it inside `spawn_blocking`. `on_window` returning
/// `false` stops the decode early (the job was cancelled, or the consumer's channel
/// closed), which is the difference between cancelling a 2-hour transcription and
/// waiting for it.
///
/// `on_progress` is called with `(decoded_ms, total_ms_estimate)` roughly once per
/// window; `total_ms_estimate` is `None` for streams whose duration the container
/// does not declare, in which case the UI shows elapsed time instead of a bar.
pub fn decode_windows(
    path: &Path,
    mut on_window: impl FnMut(Window) -> bool,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<(), MediaError> {
    let file = std::fs::File::open(path).map_err(|e| MediaError::Io(e.to_string()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions {
                enable_gapless: true,
                ..Default::default()
            },
            &MetadataOptions::default(),
        )
        .map_err(map_symphonia_error)?;
    let mut format = probed.format;

    // The first track with a real codec. A video file's tracks include the video
    // stream, whose codec is one symphonia has no decoder for, so "first track" is
    // wrong and "first track we can build a decoder for" is what this is.
    let mut selected = None;
    for track in format.tracks() {
        if track.codec_params.codec == CODEC_TYPE_NULL {
            continue;
        }
        if let Ok(decoder) =
            symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())
        {
            selected = Some((track.id, track.codec_params.clone(), decoder));
            break;
        }
    }
    let Some((track_id, params, mut decoder)) = selected else {
        return Err(MediaError::NoAudioTrack);
    };

    // Declared duration, when the container carries one: `n_frames` is in the
    // track's own timebase, so it converts through the track's sample rate, not the
    // target rate.
    let total_ms = match (params.n_frames, params.sample_rate) {
        (Some(frames), Some(rate)) if rate > 0 => Some(frames * 1000 / u64::from(rate)),
        _ => None,
    };

    let mut acc = WindowAccumulator::new();
    let mut resampler = Resampler::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // `next_packet` reports end-of-stream as an IO error with kind
            // `UnexpectedEof`; anything else is a real failure.
            Err(SymphoniaError::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(SymphoniaError::ResetRequired) => {
                // A chained/streamed source changed tracks. Everything decoded so
                // far is still valid; stopping here beats guessing at the new
                // track's alignment.
                break;
            }
            Err(e) => return Err(map_symphonia_error(e)),
        };
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // A corrupt packet mid-file is normal in real recordings; skipping it
            // loses a few ms rather than the whole job.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(map_symphonia_error(e)),
        };

        let spec = *decoded.spec();
        let buf = sample_buf.get_or_insert_with(|| {
            SampleBuffer::<f32>::new(decoded.capacity() as u64, spec)
        });
        buf.copy_interleaved_ref(decoded);
        let channels = spec.channels.count().max(1);
        let mono = downmix(buf.samples(), channels);
        resampler.push(&mono, spec.rate, &mut acc);

        while let Some(window) = acc.take_ready() {
            on_progress(window.end_ms(), total_ms);
            if !on_window(window.into_window()) {
                return Ok(());
            }
        }
    }

    // Whatever is left after the last packet is a final, short window. Dropping it
    // would silently truncate the subtitles of every file whose length is not a
    // multiple of the window size — i.e. all of them.
    if let Some(window) = acc.take_remainder() {
        on_progress(window.end_ms(), total_ms);
        on_window(window.into_window());
    }
    Ok(())
}

/// Map symphonia's error taxonomy onto ours. `Unsupported` is the interesting one:
/// it is the answer to "why did nothing happen when I picked that file", so it must
/// not be flattened into a generic decode failure.
fn map_symphonia_error(e: SymphoniaError) -> MediaError {
    match e {
        SymphoniaError::Unsupported(what) => MediaError::Unsupported(what.to_string()),
        SymphoniaError::IoError(io) => MediaError::Io(io.to_string()),
        other => MediaError::Decode(other.to_string()),
    }
}

/// Average N interleaved channels down to one. Averaging rather than taking the
/// left channel: a film's dialogue lives in the centre channel, which a left-only
/// pick would discard entirely.
fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Rate conversion to [`TARGET_RATE`] by box-averaging the source samples each
/// output sample spans.
///
/// Fractional-position state persists across calls, so a stream whose packets are
/// not a whole number of output samples long does not drift: 48 kHz → 16 kHz is
/// exact, but 44.1 kHz → 16 kHz is 2.75625 input samples per output sample, and a
/// per-packet reset would accumulate an audible offset over an hour — which shows
/// up as subtitles that creep steadily out of sync.
struct Resampler {
    /// Input samples not yet consumed by an output sample.
    carry: Vec<f32>,
    /// Fractional source position of the next output sample, relative to `carry[0]`.
    position: f64,
    /// The source rate seen so far. A change mid-stream restarts the box.
    rate: u32,
}

impl Resampler {
    fn new() -> Self {
        Self {
            carry: Vec::new(),
            position: 0.0,
            rate: 0,
        }
    }

    /// Feed `mono` (at `rate`) in, pushing 16 kHz samples into `acc`.
    fn push(&mut self, mono: &[f32], rate: u32, acc: &mut WindowAccumulator) {
        if rate == 0 {
            return;
        }
        if rate != self.rate {
            self.carry.clear();
            self.position = 0.0;
            self.rate = rate;
        }
        if rate == TARGET_RATE {
            for &s in mono {
                acc.push(to_i16(s));
            }
            return;
        }

        self.carry.extend_from_slice(mono);
        let step = f64::from(rate) / f64::from(TARGET_RATE);
        let mut consumed = 0usize;
        loop {
            let start = self.position;
            let end = start + step;
            // Need the whole span present before emitting, or the average is taken
            // over a truncated box and the output dips at every packet boundary.
            if end.ceil() as usize > self.carry.len() {
                break;
            }
            let lo = start.floor() as usize;
            let hi = (end.ceil() as usize).min(self.carry.len()).max(lo + 1);
            let span = &self.carry[lo..hi];
            let avg = span.iter().sum::<f32>() / span.len() as f32;
            acc.push(to_i16(avg));
            self.position = end;
            consumed = lo;
        }
        // Drop what no future output sample can reference, keeping `position`
        // relative to the new `carry[0]`.
        if consumed > 0 {
            self.carry.drain(..consumed);
            self.position -= consumed as f64;
        }
    }
}

/// Clamp a float sample into i16. `32767.0` (not `32768.0`) so full-scale positive
/// does not wrap.
fn to_i16(v: f32) -> i16 {
    (v * 32767.0).clamp(-32768.0, 32767.0) as i16
}

/// A window under construction, plus the silence bookkeeping that decides where it
/// ends.
struct WindowAccumulator {
    samples: Vec<i16>,
    /// Absolute offset (in 16 kHz samples) of `samples[0]` from media start.
    start_sample: u64,
    /// Sum of squares per [`SILENCE_FRAME_MS`] frame, for the quietest-frame hunt.
    frame_energy: Vec<f64>,
    /// Running energy of the in-progress frame, and how many samples are in it.
    current_energy: f64,
    current_count: u32,
    ready: Vec<PendingWindow>,
}

/// A window whose cut point has been chosen.
struct PendingWindow {
    offset_ms: u64,
    samples: Vec<i16>,
}

impl PendingWindow {
    fn end_ms(&self) -> u64 {
        self.offset_ms + samples_to_ms(self.samples.len() as u64)
    }

    fn into_window(self) -> Window {
        Window {
            offset_ms: self.offset_ms,
            duration_ms: samples_to_ms(self.samples.len() as u64),
            wav: encode_wav(&self.samples),
        }
    }
}

const fn samples_per(secs: u32) -> usize {
    (TARGET_RATE * secs) as usize
}

fn samples_to_ms(samples: u64) -> u64 {
    samples * 1000 / u64::from(TARGET_RATE)
}

impl WindowAccumulator {
    fn new() -> Self {
        Self {
            samples: Vec::with_capacity(samples_per(WINDOW_SECS + CUT_SEARCH_SECS)),
            start_sample: 0,
            frame_energy: Vec::new(),
            current_energy: 0.0,
            current_count: 0,
            ready: Vec::new(),
        }
    }

    fn push(&mut self, sample: i16) {
        self.samples.push(sample);
        let v = f64::from(sample) / 32768.0;
        self.current_energy += v * v;
        self.current_count += 1;
        if self.current_count as usize >= frame_len() {
            self.frame_energy
                .push(self.current_energy / f64::from(self.current_count));
            self.current_energy = 0.0;
            self.current_count = 0;
        }
        if self.samples.len() >= samples_per(WINDOW_SECS + CUT_SEARCH_SECS) {
            self.cut();
        }
    }

    /// Split at the quietest frame inside the search band and stage the head.
    fn cut(&mut self) {
        let cut = self.quietest_cut();
        let tail = self.samples.split_off(cut);
        let head = std::mem::replace(&mut self.samples, tail);
        let offset_ms = samples_to_ms(self.start_sample);
        self.start_sample += head.len() as u64;
        self.ready.push(PendingWindow {
            offset_ms,
            samples: head,
        });
        // Frame energies belong to the head that just left; the tail re-measures.
        self.frame_energy.clear();
        self.current_energy = 0.0;
        self.current_count = 0;
    }

    /// Index of the sample to cut at: the start of the lowest-energy frame in
    /// `[WINDOW_SECS - CUT_SEARCH_SECS, WINDOW_SECS + CUT_SEARCH_SECS]`, falling
    /// back to the nominal point when the band holds no measured frame.
    fn quietest_cut(&self) -> usize {
        let nominal = samples_per(WINDOW_SECS);
        let lo_sample = samples_per(WINDOW_SECS - CUT_SEARCH_SECS);
        let hi_sample = samples_per(WINDOW_SECS + CUT_SEARCH_SECS).min(self.samples.len());
        let lo_frame = lo_sample / frame_len();
        let hi_frame = (hi_sample / frame_len()).min(self.frame_energy.len());
        if lo_frame >= hi_frame {
            return nominal.min(self.samples.len());
        }
        let mut best = lo_frame;
        let mut best_energy = f64::MAX;
        for (i, energy) in self
            .frame_energy
            .iter()
            .enumerate()
            .take(hi_frame)
            .skip(lo_frame)
        {
            if *energy < best_energy {
                best_energy = *energy;
                best = i;
            }
        }
        // Cut in the MIDDLE of the quiet frame, so neither neighbour keeps the
        // trailing breath of the other.
        ((best * frame_len()) + frame_len() / 2).min(self.samples.len())
    }

    fn take_ready(&mut self) -> Option<PendingWindow> {
        if self.ready.is_empty() {
            None
        } else {
            Some(self.ready.remove(0))
        }
    }

    /// The final partial window. `None` when nothing is left, or when what is left
    /// is too short to hold speech (a stray few ms after the last cut) — whisper
    /// answers a 20 ms window with a hallucinated word surprisingly often.
    fn take_remainder(&mut self) -> Option<PendingWindow> {
        if let Some(pending) = self.take_ready() {
            return Some(pending);
        }
        if self.samples.len() < samples_per(1) / 4 {
            return None;
        }
        let offset_ms = samples_to_ms(self.start_sample);
        let samples = std::mem::take(&mut self.samples);
        self.start_sample += samples.len() as u64;
        Some(PendingWindow { offset_ms, samples })
    }
}

fn frame_len() -> usize {
    (TARGET_RATE * SILENCE_FRAME_MS / 1000) as usize
}

/// Wrap 16 kHz mono i16 PCM in a WAV header. `hound` writes into a `Cursor`, so no
/// temporary file touches the disk on the way to the engine.
fn encode_wav(samples: &[i16]) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    // Every failure below is an in-memory write, which cannot fail in practice; the
    // fallback is an empty buffer rather than a panic in a worker thread.
    let Ok(mut writer) = hound::WavWriter::new(&mut cursor, spec) else {
        return Vec::new();
    };
    for &s in samples {
        if writer.write_sample(s).is_err() {
            return Vec::new();
        }
    }
    if writer.finalize().is_err() {
        return Vec::new();
    }
    cursor.into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tone at `hz`, `secs` long, at `rate`.
    fn tone(hz: f32, secs: f32, rate: u32) -> Vec<f32> {
        let n = (secs * rate as f32) as usize;
        (0..n)
            .map(|i| {
                (i as f32 / rate as f32 * hz * std::f32::consts::TAU).sin() * 0.5
            })
            .collect()
    }

    #[test]
    fn downmix_averages_channels() {
        // Two frames of stereo: (1.0, 0.0) and (0.5, 0.5).
        let mono = downmix(&[1.0, 0.0, 0.5, 0.5], 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn downmix_passes_mono_through() {
        assert_eq!(downmix(&[0.25, -0.25], 1), vec![0.25, -0.25]);
    }

    #[test]
    fn resampler_hits_the_target_rate_within_a_sample() {
        let mut acc = WindowAccumulator::new();
        let mut r = Resampler::new();
        r.push(&tone(440.0, 1.0, 48_000), 48_000, &mut acc);
        let produced = acc.samples.len();
        assert!(
            produced.abs_diff(TARGET_RATE as usize) <= 2,
            "1 s of 48 kHz should resample to ~16000 samples, got {produced}"
        );
    }

    #[test]
    fn resampler_does_not_drift_across_packet_boundaries() {
        // 44.1 kHz is the non-integer ratio: 10 packets of 0.5 s must still land on
        // 5 s of output, not 5 s ± a packet's worth of accumulated rounding.
        let mut acc = WindowAccumulator::new();
        let mut r = Resampler::new();
        for _ in 0..10 {
            r.push(&tone(440.0, 0.5, 44_100), 44_100, &mut acc);
        }
        let expected = TARGET_RATE as usize * 5;
        let produced = acc.samples.len() + acc.ready.iter().map(|w| w.samples.len()).sum::<usize>();
        assert!(
            produced.abs_diff(expected) <= 4,
            "5 s at 44.1 kHz should be ~{expected} samples, got {produced}"
        );
    }

    #[test]
    fn passthrough_when_already_at_target_rate() {
        let mut acc = WindowAccumulator::new();
        let mut r = Resampler::new();
        r.push(&[0.5, -0.5], TARGET_RATE, &mut acc);
        assert_eq!(acc.samples.len(), 2);
    }

    #[test]
    fn windows_cut_at_the_quiet_gap_not_the_nominal_point() {
        let mut acc = WindowAccumulator::new();
        // Loud everywhere except a 300 ms gap two seconds PAST the nominal cut.
        let gap_start = samples_per(WINDOW_SECS + 2);
        let gap_end = gap_start + samples_per(1) / 3;
        for i in 0..samples_per(WINDOW_SECS + CUT_SEARCH_SECS) {
            let v = if i >= gap_start && i < gap_end { 0 } else { 12_000 };
            acc.push(v);
        }
        let window = acc.take_ready().expect("a full window should be staged");
        let cut = window.samples.len();
        assert!(
            cut > gap_start && cut < gap_end,
            "cut {cut} should land inside the quiet gap {gap_start}..{gap_end}"
        );
    }

    #[test]
    fn window_offsets_are_contiguous_and_absolute() {
        let mut acc = WindowAccumulator::new();
        for _ in 0..samples_per((WINDOW_SECS + CUT_SEARCH_SECS) * 2 + 1) {
            acc.push(1000);
        }
        let first = acc.take_ready().expect("first window");
        let second = acc.take_ready().expect("second window");
        assert_eq!(first.offset_ms, 0);
        assert_eq!(second.offset_ms, first.end_ms());
    }

    #[test]
    fn remainder_is_emitted_but_a_sliver_is_not() {
        let mut acc = WindowAccumulator::new();
        for _ in 0..samples_per(2) {
            acc.push(500);
        }
        assert!(acc.take_remainder().is_some(), "2 s tail is real audio");

        let mut sliver = WindowAccumulator::new();
        for _ in 0..10 {
            sliver.push(500);
        }
        assert!(
            sliver.take_remainder().is_none(),
            "a 10-sample tail is not worth a hallucinated word"
        );
    }

    #[test]
    fn encoded_window_is_a_readable_16k_mono_wav() {
        let pcm: Vec<i16> = (0..1600).map(|i| (i % 100) as i16 * 100).collect();
        let wav = encode_wav(&pcm);
        let reader = hound::WavReader::new(std::io::Cursor::new(wav)).expect("valid WAV");
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, TARGET_RATE);
        assert_eq!(reader.len() as usize, pcm.len());
    }

    #[test]
    fn media_extension_gate_is_case_insensitive_and_closed() {
        assert!(is_media_path(Path::new("/a/b/Talk.MP4")));
        assert!(is_media_path(Path::new("/a/b/talk.mkv")));
        assert!(!is_media_path(Path::new("/a/b/talk.txt")));
        assert!(!is_media_path(Path::new("/etc/passwd")));
    }

    #[test]
    fn missing_file_is_an_io_error_not_a_panic() {
        let err = decode_windows(Path::new("/nope/nothing.mp4"), |_| true, |_, _| {})
            .expect_err("a missing file must fail");
        assert!(matches!(err, MediaError::Io(_)));
    }

    #[test]
    fn a_real_wav_decodes_end_to_end_into_windows() {
        // The one container we can synthesize in a unit test without fixtures. It
        // still exercises probe → decode → downmix → resample → window.
        let dir = std::env::temp_dir().join(format!("ryu-subtitles-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("tone.wav");
        {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 44_100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::create(&path, spec).expect("write fixture");
            for s in tone(440.0, 2.0, 44_100) {
                let v = (s * 20_000.0) as i16;
                w.write_sample(v).expect("l");
                w.write_sample(v).expect("r");
            }
            w.finalize().expect("finalize");
        }

        let mut windows = Vec::new();
        decode_windows(
            &path,
            |w| {
                windows.push(w);
                true
            },
            |_, _| {},
        )
        .expect("decode");
        std::fs::remove_file(&path).ok();

        assert_eq!(windows.len(), 1, "2 s is one short window");
        assert_eq!(windows[0].offset_ms, 0);
        assert!(
            windows[0].duration_ms.abs_diff(2000) < 50,
            "window should be ~2000 ms, got {}",
            windows[0].duration_ms
        );
    }
}
