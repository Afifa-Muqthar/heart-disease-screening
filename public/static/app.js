const form = document.querySelector("#screening-form");
const resultCard = document.querySelector(".result-card");
const resultContent = document.querySelector("#result-content");
const formMessage = document.querySelector("#form-message");
const submitButton = document.querySelector("#submit-button");

function renderEmpty() {
  resultContent.innerHTML = '<div class="empty-mark" aria-hidden="true"><svg viewBox="0 0 48 48"><circle cx="24" cy="24" r="18"/><path d="M24 14v20M14 24h20"/></svg></div><p class="empty-title" id="result-title">Your estimate will appear here.</p><p class="empty-copy">Complete the five details, then select “See your estimate”.</p>';
  resultCard.dataset.state = "empty";
}

function showMessage(message, state = "error") {
  resultContent.innerHTML = `<div class="state-symbol" aria-hidden="true">${state === "loading" ? '<span class="spinner"></span>' : '<svg viewBox="0 0 24 24"><path d="M12 3 2.8 20h18.4L12 3Z"/><path d="M12 9v5m0 3h.01"/></svg>'}</div><p class="empty-title" id="result-title">${state === "loading" ? "Comparing your answers…" : "We could not make an estimate."}</p><p class="empty-copy"></p>`;
  resultContent.querySelector(".empty-copy").textContent = message;
  resultCard.dataset.state = state;
}

function makeTextElement(tagName, className, text) {
  const element = document.createElement(tagName);
  element.className = className;
  element.textContent = text;
  return element;
}

function showEstimate(data) {
  if (data.band === "unavailable") {
    showMessage(data.copy);
    return;
  }
  const [description, referenceLine = ""] = data.copy.split("\n", 2);
  const bandName = data.band_name;
  resultContent.replaceChildren(
    makeTextElement("p", "band-kicker", "DATASET SIMILARITY"),
    makeTextElement("h2", "band-title", bandName),
    makeTextElement("p", "band-copy", description),
    makeTextElement("p", "reference-line", referenceLine)
  );
  resultContent.querySelector(".band-title").id = "result-title";
  const scale = document.createElement("div");
  scale.className = "band-scale";
  scale.setAttribute("aria-label", "Lower to higher similarity");
  for (const [className, name] of [["lower", "Lower"], ["middle", "Intermediate"], ["higher", "Higher"]]) {
    const segment = document.createElement("span");
    segment.className = `scale-segment ${className}${bandName === name ? " active" : ""}`;
    scale.append(segment);
  }
  for (const [className, label] of [["low", "LOWER"], ["mid", "INTERMEDIATE"], ["high", "HIGHER"]]) {
    scale.append(makeTextElement("span", `scale-label ${className}`, label));
  }
  resultContent.append(scale);
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

