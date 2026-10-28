const form = document.querySelector('#match-form');
const status = document.querySelector('#status');
const results = document.querySelector('#results');

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  })[char]);
}

function list(items) {
  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
}

function render(data) {
  results.innerHTML = data.matches.map((match, index) => `
    <article class="card ${index === 0 ? 'best' : ''}">
      <div class="card-top">
        <span class="rank">${index === 0 ? 'BEST MATCH' : `OPTION ${index + 1}`}</span>
        <span class="score">${match.match_percent}% fit</span>
      </div>
      <h2>${escapeHtml(match.title)}</h2>
      <p>${escapeHtml(match.description)}</p>
      <div class="meta"><span>${match.minutes} min</span><span>${escapeHtml(match.energy)} energy</span><span>${escapeHtml(match.accessibility.join(' · '))}</span></div>
      <h3>Three-step plan</h3>${list(match.steps)}
      <h3>Bring</h3>${list(match.bring)}
      <p class="safety"><strong>Safety:</strong> ${escapeHtml(match.safety)}</p>
    </article>
  `).join('');
  status.innerHTML = `<strong>${data.eligible_count}</strong> catalogue activities passed every hard constraint. ${escapeHtml(data.privacy)}`;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  results.innerHTML = '';
  status.textContent = 'Running the local model…';
  const values = new FormData(form);
  const payload = Object.fromEntries(values.entries());
  payload.minutes = Number(payload.minutes);
  try {
    const response = await fetch('/api/match', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to match activities.');
    render(data);
  } catch (error) {
    status.textContent = error.message;
  }
});

if (new URLSearchParams(window.location.search).get('demo') === '1') {
  form.elements.note.value = 'quiet trees, sketching, and mindful observation';
  form.requestSubmit();
}

