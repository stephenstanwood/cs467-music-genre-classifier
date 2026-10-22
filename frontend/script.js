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
const trackDescription = document.getElementById("track-description");

let audioUrl = null; // Temporary link used by the audio player.
let working = false; // Prevents starting two requests at once.

// When a new file is chosen, update the player and track name.
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  errorMessage.hidden = true;
  spectrogramImage.hidden = true;
  spectrogramImage.removeAttribute("src");
  analyzeButton.disabled = !file || working;

  // Free the old temporary link before making another one.
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = null;
  audioPlayer.hidden = !file;

  if (!file) {
    audioPlayer.removeAttribute("src");
    trackName.textContent = "No track selected";
    trackDescription.textContent = "Demo scores · Not model predictions";
    statusText.textContent = "Choose an audio file and select Analyze track.";
    return;
  }

  // The player uses the local file. It does not need the Python server.
  audioUrl = URL.createObjectURL(file);
  audioPlayer.src = audioUrl;
  trackName.textContent = file.name;
  trackDescription.textContent = "Selected audio · Genre scores are demo values";
  statusText.textContent = "Ready to analyze: " + file.name;
});

// When Analyze is clicked, send the audio to Python.
analyzeButton.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file || working) return;

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
    const imageUrl = URL.createObjectURL(png);
    spectrogramImage.onload = () => URL.revokeObjectURL(imageUrl);
    spectrogramImage.src = imageUrl;
    spectrogramImage.hidden = false;
    statusText.textContent = "Mel spectrogram generated for " + file.name + ".";
  } catch (error) {
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
