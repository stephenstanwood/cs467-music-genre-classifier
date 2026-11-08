"use strict";

// Get the page elements that need to change when an audio file is selected.
const fileInput = document.getElementById("audio-file");
const analyzeButton = document.getElementById("analyze-button");
const audioPlayer = document.getElementById("audio-player");
const spectrogramImage = document.getElementById("spectrogram");
const statusText = document.getElementById("spectrogram-status");
const errorMessage = document.getElementById("error-message");
const loadingMessage = document.getElementById("loading-message");
const trackName = document.getElementById("track-name");

const waveformPlayer = document.getElementById("waveform-player");
const waveform = document.getElementById("waveform");
const waveformSeek = document.getElementById("waveform-seek");
const playToggle = document.getElementById("play-toggle");
const waveformContext = waveform.getContext("2d");
let peaks = [];

// Estimate bass, midrange, and treble levels across the full track.
function getFrequencyPeaks(audio, barCount) {
  const samples = audio.getChannelData(0);
  const bassBlend = 1 - Math.exp(-2 * Math.PI * 250 / audio.sampleRate);
  const midBlend = 1 - Math.exp(-2 * Math.PI * 2500 / audio.sampleRate);

  let bass = 0;
  let bassAndMids = 0;
  let loudestPeak = 0;

  const bars = Array.from({ length: barCount }, (_, index) => {
    const start = Math.floor(index * samples.length / barCount);
    const end = Math.floor((index + 1) * samples.length / barCount);

    let bassPeak = 0;
    let midPeak = 0;
    let treblePeak = 0;

    for (let i = start; i < end; i++) {
      const sample = samples[i];

      // Smooth the signal to estimate the lower frequencies.
      bass += bassBlend * (sample - bass);
      bassAndMids += midBlend * (sample - bassAndMids);

      // Subtract the lower bands to estimate the mids and treble.
      bassPeak = Math.max(bassPeak, Math.abs(bass));
      midPeak = Math.max(midPeak, Math.abs(bassAndMids - bass));
      treblePeak = Math.max(treblePeak, Math.abs(sample - bassAndMids));
    }

    loudestPeak = Math.max(loudestPeak, bassPeak, midPeak, treblePeak);
    return [bassPeak, midPeak, treblePeak];
  });

  // Scale all three bands together so their relative levels stay visible.
  const scale = Math.max(loudestPeak, 0.01);
  return bars.map((bands) => bands.map((level) => level / scale));
}

// Draw the frequency bands and show the current playback position.
function drawWaveform() {
  const { width, height } = waveform;
  const duration = audioPlayer.duration;
  const progress = Number.isFinite(duration) && duration > 0
    ? Math.min(audioPlayer.currentTime / duration, 1)
    : 0;

  waveformContext.clearRect(0, 0, width, height);

  const spacing = width / peaks.length;
  const colors = ["#566bff", "#779cff", "#b4d8ff"];

  // Layer the frequency bands from deep blue to pale blue.
  peaks.forEach((bands, index) => {
    bands.forEach((level, band) => {
      if (level < 0.01) return;

      const barHeight = level * (height - 6);
      waveformContext.fillStyle = colors[band];
      waveformContext.fillRect(
        index * spacing,
        (height - barHeight) / 2,
        spacing * 0.8,
        barHeight
      );
    });
  });

  waveformContext.fillStyle = "#e6edff";
  const playheadX = Math.min(Math.round(progress * width), width - 2);
  waveformContext.fillRect(playheadX, 0, 2, height);

  waveformSeek.value = Math.round(progress * Number(waveformSeek.max));
}

async function loadWaveform(file, previewUrl) {
  if (!waveformContext || !window.AudioContext) return;

  try {
    const context = new AudioContext();
    let decoded;
    try {
      decoded = await context.decodeAudioData(await file.arrayBuffer());
    } finally {
      await context.close();
    }

    if (audioUrl !== previewUrl) return;
    peaks = getFrequencyPeaks(decoded, Math.min(600, decoded.length));
    waveform.hidden = false;
    updateWaveformProgress();
    waveformPlayer.hidden = false;
    setUploadSuccess(true);
    audioPlayer.hidden = true;
  } catch {
    // The native player remains available if waveform decoding fails.
  }
}

function updateWaveformProgress() {
  const duration = audioPlayer.duration;
  waveformSeek.disabled = !Number.isFinite(duration) || duration <= 0;
  waveformSeek.value = waveformSeek.disabled
    ? 0
    : audioPlayer.currentTime / duration * Number(waveformSeek.max);
  if (peaks.length) drawWaveform();
}

playToggle.addEventListener("click", async () => {
  if (!audioPlayer.paused) {
    audioPlayer.pause();
    return;
  }
  const previewUrl = audioUrl;
  try {
    await audioPlayer.play();
  } catch {
    if (audioUrl === previewUrl) {
      waveformPlayer.hidden = true;
      audioPlayer.hidden = false;
    }
  }
});

for (const event of ["play", "pause", "ended"]) {
  audioPlayer.addEventListener(event, () => {
    const playing = !audioPlayer.paused && !audioPlayer.ended;
    playToggle.dataset.playing = String(playing);
    playToggle.setAttribute("aria-label", playing ? "Pause track" : "Play track");
  });
}
for (const event of ["durationchange", "timeupdate"]) {
  audioPlayer.addEventListener(event, updateWaveformProgress);
}
waveformSeek.addEventListener("input", () => {
  audioPlayer.currentTime = Number(waveformSeek.value) / Number(waveformSeek.max) * audioPlayer.duration;
  drawWaveform();
});

const uploadZone = document.getElementById("upload-zone");
const uploadSymbol = document.getElementById("upload-symbol");
const uploadTitle = document.getElementById("upload-title");
const uploadInstruction = document.getElementById("upload-instruction");

function setUploadSuccess(success) {
  uploadZone.classList.toggle("is-uploaded", success);
  uploadSymbol.textContent = success ? "✓" : "♫";
  uploadTitle.textContent = success ? "Audio uploaded successfully" : "Choose an audio file";
  uploadInstruction.textContent = success ? "Click to choose another file" : "Click to browse your files";
}

let audioUrl = null; // Temporary link used by the audio player.
let working = false; // Prevents starting two requests at once.

// The native player can preview some files even when waveform decoding fails.
audioPlayer.addEventListener("loadedmetadata", () => {
  if (audioUrl && audioPlayer.currentSrc === audioUrl) setUploadSuccess(true);
});

// When a new file is chosen, update the player and track name.
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  setUploadSuccess(false);
  errorMessage.hidden = true;
  spectrogramImage.hidden = true;
  spectrogramImage.removeAttribute("src");
  analyzeButton.disabled = !file || working;

  // Free the old temporary link before making another one.
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = null;
  audioPlayer.hidden = !file;
  waveformPlayer.hidden = true;
  peaks = [];
  waveformSeek.value = 0;
  playToggle.dataset.playing = "false";
  playToggle.setAttribute("aria-label", "Play track");

  if (!file) {
    audioPlayer.removeAttribute("src");
    trackName.textContent = "No track selected";
    statusText.textContent = "Choose an audio file and select Analyze track.";
    return;
  }

  // The player uses the local file. It does not need the Python server.
  audioUrl = URL.createObjectURL(file);
  audioPlayer.src = audioUrl;
  void loadWaveform(file, audioUrl);
  trackName.textContent = file.name;
  statusText.textContent = "Ready to analyze: " + file.name;
});

// When Analyze is clicked, send the audio to Python.
analyzeButton.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file || working) return;

  const previewUrl = audioUrl;
  working = true;
  analyzeButton.disabled = true;
  loadingMessage.hidden = false;
  errorMessage.hidden = true;
  spectrogramImage.hidden = true;
  statusText.textContent = "Python is creating the Mel spectrogram…";

  // FormData lets us upload the file without converting it ourselves.
  const formData = new FormData();
  formData.append("file", file);

  try {
    // FastAPI receives the file at /spectrogram and returns a PNG image.
    const response = await fetch("/spectrogram", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      let message = "The server could not process this audio file.";
      try {
        const details = await response.json();
        if (typeof details.detail === "string") message = details.detail;
      } catch (_) { /* Keep the simple message if no JSON was returned. */ }
      throw new Error(message);
    }

    // Turn the returned PNG into a temporary URL for the image element.
    const png = await response.blob();
    if (audioUrl !== previewUrl) return;
    const imageUrl = URL.createObjectURL(png);
    spectrogramImage.onload = () => URL.revokeObjectURL(imageUrl);
    spectrogramImage.src = imageUrl;
    spectrogramImage.hidden = false;
    statusText.textContent = "Mel spectrogram generated for " + file.name + ".";
  } catch (error) {
    if (audioUrl !== previewUrl) return;
    errorMessage.hidden = false;
    errorMessage.querySelector("p").textContent = error.message;
    statusText.textContent = "Spectrogram generation failed. Make sure the Python server is running.";
  } finally {
    // Restore the button and hide the loading indicator.
    working = false;
    loadingMessage.hidden = true;
    analyzeButton.disabled = !fileInput.files.length;
  }
});
