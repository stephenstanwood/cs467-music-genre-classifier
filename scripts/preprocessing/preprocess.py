"""
Shared audio preprocessing for the genre classifier.
 
Training data and user uploads MUST both go through this file, so the model
always sees numbers produced the exact same way.
 
Main entry points:
    load_audio(path)            -> mono audio at SAMPLE_RATE
    split_into_segments(y)      -> list of 3-second pieces
    extract_features(segment)   -> dict of the 57 GTZAN-style features
    mel_spectrogram(segment)    -> 2D array for the CNN (next experiment)
    features_for_file(path)     -> one row of 57 features per segment
"""
 
import numpy as np
import librosa
 
SAMPLE_RATE = 22050        # Hz, mono (matches GTZAN)
SEGMENT_SECONDS = 3        # length of each piece
N_MFCC = 20                # MFCC coefficients
N_MELS = 128               # mel bands for the spectrogram CNN
N_FFT = 2048
HOP_LENGTH = 512
 
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS
 
# Column order matches GTZAN's features_3_sec.csv so the two are comparable.
FEATURE_COLUMNS = (
    [f"{name}_{stat}"
     for name in ["chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth",
                  "rolloff", "zero_crossing_rate", "harmony", "perceptr"]
     for stat in ["mean", "var"]]
    + ["tempo"]
    + [f"mfcc{i}_{stat}" for i in range(1, N_MFCC + 1) for stat in ["mean", "var"]]
)
MFCC_COLUMNS = [c for c in FEATURE_COLUMNS if c.startswith("mfcc")]  # the 40-column subset
 
assert len(FEATURE_COLUMNS) == 57
 
 
def load_audio(path, duration=None):
    """Load any audio file (WAV, MP3, ...) as mono at SAMPLE_RATE.
    Raises ValueError if the file is unreadable or silent."""
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=duration)
    except Exception as e:
        raise ValueError(f"Could not read audio file {path}: {e}") from e
    if y.size == 0 or not np.any(y):
        raise ValueError(f"Audio file is empty or silent: {path}")
    return y
 
 
def split_into_segments(y, min_fraction=0.5):
    """Cut audio into back-to-back SEGMENT_SECONDS pieces.
    A short final piece is zero-padded if it's at least min_fraction long,
    otherwise dropped. Raises ValueError if the clip is too short overall."""
    segments = []
    for start in range(0, len(y), SEGMENT_SAMPLES):
        piece = y[start:start + SEGMENT_SAMPLES]
        if len(piece) < SEGMENT_SAMPLES:
            if len(piece) < SEGMENT_SAMPLES * min_fraction:
                break
            piece = librosa.util.fix_length(piece, size=SEGMENT_SAMPLES)
        segments.append(piece)
    if not segments:
        raise ValueError(f"Clip too short: need at least "
                         f"{SEGMENT_SECONDS * min_fraction:.1f} seconds of audio")
    return segments
 
 
def _mean_var(name, values, out):
    out[f"{name}_mean"] = float(np.mean(values))
    out[f"{name}_var"] = float(np.var(values))
 
 
try:  # librosa >= 0.10
    from librosa.feature.rhythm import tempo as _librosa_tempo
except ImportError:  # older librosa
    from librosa.beat import tempo as _librosa_tempo
 
 
def _tempo(y):
    t = _librosa_tempo(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    return float(np.atleast_1d(t)[0])
 
 
def extract_features(segment):
    """Return the 57 GTZAN-style features for one segment, as a dict
    in FEATURE_COLUMNS order."""
    sr, kw = SAMPLE_RATE, dict(n_fft=N_FFT, hop_length=HOP_LENGTH)
    f = {}
    _mean_var("chroma_stft", librosa.feature.chroma_stft(y=segment, sr=sr, **kw), f)
    _mean_var("rms", librosa.feature.rms(y=segment, frame_length=N_FFT, hop_length=HOP_LENGTH), f)
    _mean_var("spectral_centroid", librosa.feature.spectral_centroid(y=segment, sr=sr, **kw), f)
    _mean_var("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=segment, sr=sr, **kw), f)
    _mean_var("rolloff", librosa.feature.spectral_rolloff(y=segment, sr=sr, **kw), f)
    _mean_var("zero_crossing_rate",
              librosa.feature.zero_crossing_rate(segment, frame_length=N_FFT, hop_length=HOP_LENGTH), f)
    harmonic, percussive = librosa.effects.hpss(segment)
    _mean_var("harmony", harmonic, f)
    _mean_var("perceptr", percussive, f)
    f["tempo"] = _tempo(segment)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, **kw)
    for i in range(N_MFCC):
        _mean_var(f"mfcc{i + 1}", mfcc[i], f)
    return {c: f[c] for c in FEATURE_COLUMNS}
 
 
def mel_spectrogram(segment):
    """Log-scaled mel-spectrogram (N_MELS x frames) for the CNN experiment."""
    mel = librosa.feature.melspectrogram(y=segment, sr=SAMPLE_RATE, n_fft=N_FFT,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
 
 
def features_for_file(path, duration=None):
    """Full pipeline for one file: load -> segment -> 57 features per segment.
    Returns a list of dicts, one per segment, each with a 'segment' index.
    This is the function the API should call on user uploads."""
    y = load_audio(path, duration=duration)
    rows = []
    for i, seg in enumerate(split_into_segments(y)):
        row = {"segment": i}
        row.update(extract_features(seg))
        rows.append(row)
    return rows
 