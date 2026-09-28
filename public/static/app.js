const form = document.querySelector("#screening-form");
const resultCard = document.querySelector(".result-card");
const resultContent = document.querySelector("#result-content");
const formMessage = document.querySelector("#form-message");
const submitButton = document.querySelector("#submit-button");

const disclaimer = "Screening estimate from a clinic-patient dataset. Not a diagnosis, not population risk. Talk to a clinician about symptoms.";

function renderEmpty() {
  resultContent.innerHTML = '<div class="empty-mark" aria-hidden="true"><svg viewBox="0 0 48 48"><circle cx="24" cy="24" r="18"/><path d="M24 14v20M14 24h20"/></svg></div><p class="empty-title" id="result-title">Your estimate will appear here.</p><p class="empty-copy">Complete the five details, then select “See your estimate”.</p>';
  resultCard.dataset.state = "empty";
}

function showMessage(message, state = "error") {
  resultContent.innerHTML = `<div class="state-symbol" aria-hidden="true">${state === "loading" ? '<span class="spinner"></span>' : '<svg viewBox="0 0 24 24"><path d="M12 3 2.8 20h18.4L12 3Z"/><path d="M12 9v5m0 3h.01"/></svg>'}</div><p class="empty-title" id="result-title">${state === "loading" ? "Comparing your answers…" : "We could not make an estimate."}</p><p class="empty-copy">${message}</p>`;
  resultCard.dataset.state = state;
}

function showEstimate(data) {
  if (data.band === "unavailable") {
    showMessage(data.copy);
    return;
  }
  const [description, referenceLine = ""] = data.copy.split("\n", 2);
  const bandName = data.band_name;
  resultContent.innerHTML = `<p class="band-kicker">DATASET SIMILARITY</p><h2 class="band-title" id="result-title">${bandName}</h2><p class="band-copy">${description}</p><p class="reference-line">${referenceLine}</p><div class="band-scale" aria-label="Lower to higher similarity"><span class="scale-segment lower ${bandName === "Lower" ? "active" : ""}"></span><span class="scale-segment middle ${bandName === "Intermediate" ? "active" : ""}"></span><span class="scale-segment higher ${bandName === "Higher" ? "active" : ""}"></span><span class="scale-label low">LOWER</span><span class="scale-label mid">INTERMEDIATE</span><span class="scale-label high">HIGHER</span></div>`;
  resultCard.dataset.state = "result";
}

function selected(name) {
  return form.querySelector(`input[name="${name}"]:checked`)?.value;
}

function validate() {
  const age = form.elements.age.value;
  const bp = form.elements.trestbps.value;
  if (!age || !selected("sex") || !selected("cp") || selected("exang") === undefined || !bp) {
    formMessage.textContent = "Complete all five inputs before requesting an estimate.";
    formMessage.dataset.state = "error";
    showMessage("One or more answers are missing. Complete all five inputs and try again.");
    return null;
  }
  if (!Number.isInteger(Number(age)) || Number(age) < 18 || Number(age) > 100) {
    formMessage.textContent = "Enter an age from 18 to 100.";
    formMessage.dataset.state = "error";
    showMessage("Age must be a whole number from 18 to 100.");
    return null;
  }
  if (Number(bp) < 80 || Number(bp) > 200) {
    formMessage.textContent = "Enter blood pressure from 80 to 200 mmHg.";
    formMessage.dataset.state = "error";
    showMessage("Resting systolic blood pressure must be between 80 and 200 mmHg.");
    return null;
  }
  return {age: Number(age), sex: selected("sex"), cp: selected("cp"), exang: selected("exang") === "true", trestbps: Number(bp)};
}

form.addEventListener("input", () => {
  if (formMessage.dataset.state === "error") {
    formMessage.textContent = "Answers are used only to create this estimate and are not saved.";
    formMessage.dataset.state = "";
    renderEmpty();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = validate();
  if (!payload) return;
  formMessage.dataset.state = "";
  formMessage.textContent = "Checking the reference model…";
  submitButton.disabled = true;
  resultCard.setAttribute("aria-busy", "true");
  showMessage("Using the approved five-input model.", "loading");
  try {
    const response = await fetch("/api/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The request could not be completed.");
    showEstimate(data);
    formMessage.textContent = data.band === "unavailable" ? "No score was produced for this age." : "Estimate ready. No medical diagnosis is made.";
  } catch (error) {
    showMessage(error.message || "Check your connection and try again.");
    formMessage.textContent = "The estimate could not be completed.";
    formMessage.dataset.state = "error";
  } finally {
    submitButton.disabled = false;
    resultCard.setAttribute("aria-busy", "false");
  }
});

