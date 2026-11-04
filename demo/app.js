'use strict';
const $ = id => document.getElementById(id);
let socket, state, commands = {}, connected = false, imageUrl, reconnectTimer, presetSignature;
let camera = {azimuth: 135, elevation: -18, distance: 4}, dragging;
let mapView = {cx: 0, cy: 0, span: 8};
let suggestionIndex = -1;
const message = (text, error = false) => { $('feedback').textContent = text; $('feedback').classList.toggle('error', error); };
function send(action, extra = {}) {
  if (!connected) { message('Simulator disconnected. Reconnecting…', true); return; }
  socket.send(JSON.stringify({action, ...extra}));
}
function setConnected(value) {
  connected = value;
  $('dot').classList.toggle('online', value);
  $('connection').textContent = value ? 'Simulator connected' : 'Disconnected · reconnecting…';
  document.querySelectorAll('button').forEach(button => { button.disabled = !value; });
}
function connect() {
  socket = window.createLocalSimulator ? window.createLocalSimulator() : new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  socket.binaryType = 'blob';
  socket.onmessage = event => {
    if (event.data instanceof Blob) {
      const oldUrl = imageUrl;
      imageUrl = URL.createObjectURL(event.data);
      $('frame').src = imageUrl;
      if (oldUrl) URL.revokeObjectURL(oldUrl);
      $('loading').hidden = true;
      return;
    }
    const data = JSON.parse(event.data);
    if (data.type === 'progress') {
      setConnected(false);
      $('loading').hidden = false;
      $('loading').textContent = data.message;
      $('connection').textContent = 'Loading on your device…';
      message(data.message);
      return;
    }
    if (data.type === 'hello') {
      commands = data.commands;
      setConnected(true);
      message(window.createLocalSimulator ? 'Ready on your device. Choose a motion, then press Run.' : 'Simulator connected. Choose a motion, then press Run.');
      $('loading').hidden = true;
      updatePresets();
    } else if (data.type === 'state') {
      const changedMode = state?.mode !== data.mode;
      state = data;
      renderState(changedMode);
    } else if (data.type === 'error') {
      message(data.message, true);
    } else if (data.type === 'ack') {
      const feedback = {text: 'Motion command applied.', goal: 'Goal placed. Press Run to start, or keep exploring while running.', reset: 'Robot reset. Choose a motion, then press Run.', mode: 'Policy loaded and robot reset.', run: 'Policy running. You can change the motion at any time.', pause: 'Paused at a control boundary.', clear_goal: 'Goal cleared. Motion set to stand.'};
      if (feedback[data.action]) message(feedback[data.action]);
    }
  };
  socket.onclose = () => {
    setConnected(false);
    message('Connection lost. The view shows the last received frame.', true);
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connect, 2000);
  };
  socket.onerror = () => socket.close();
}
function currentCommands() { return commands[state?.mode || 'text'] || []; }
function updatePresets() {
  const signature = `${state?.mode}|${state?.active_text}|${state?.text}|${currentCommands().length}`;
  if (signature === presetSignature) return;
  presetSignature = signature;
  $('presets').replaceChildren();
  const available = currentCommands();
  const presets = state?.mode === 'goal' ? ['walk', 'walk while bend over', 'jog', 'jog in place', 'stand'] : ['tpose', 'apose', 'wave left hand', 'arm circles', 'clap', 'shoot a bow', 'throw a ball', 'spin', 'walk back', 'look back', 'squat', 'backward right punch', 'jog in circle', 'sprint in place', 'drop kick', 'kick left leg', 'kick right leg', 'dance around', 'side step to left', 'crouch with hands up side step to right series', 'hop on one foot', 'jump rope'];
  for (const label of presets) {
    // Preserve the requested button wording while applying the exact trained key.
    const text = label === 'backward right punch' ? 'backwards right punch' : label;
    if (!available.includes(text)) continue;
    const button = document.createElement('button');
    button.textContent = label;
    button.dataset.command = text;
    if (label !== text) button.title = `Training command: ${text}`;
    button.classList.toggle('selected', text === (state?.mode === 'goal' ? state?.active_text : state?.text));
    button.onclick = () => applyText(text);
    $('presets').append(button);
  }
  $('command-count').textContent = `${available.length.toLocaleString()} ${state?.mode === 'goal' ? 'goal' : 'training'} commands`;
  $('browse-commands').textContent = `Browse all ${available.length.toLocaleString()} commands`;
  $('preset-heading').textContent = state?.mode === 'goal' ? 'MOTION STYLES' : 'QUICK PICKS';
}
function applyText(text) {
  if (!currentCommands().includes(text)) {
    message(state?.mode === 'goal' ? 'Choose a command from the goal-reaching list.' : 'Choose a command from the allowed training text set.', true);
    return;
  }
  $('command').value = text;
  hideSuggestions();
  send('text', {text});
}
function hideSuggestions() {
  $('suggestions').hidden = true;
  $('command').setAttribute('aria-expanded', 'false');
  $('command').removeAttribute('aria-activedescendant');
  $('browse-commands').setAttribute('aria-expanded', 'false');
  suggestionIndex = -1;
}
function showSuggestions(value = $('command').value) {
  const query = value.toLowerCase().trim();
  const rank = text => text.toLowerCase() === query ? 0 : text.toLowerCase().startsWith(query) ? 1 : 2;
  const matches = currentCommands().filter(text => text.toLowerCase().includes(query)).sort((a, b) => rank(a) - rank(b));
  $('suggestions').replaceChildren();
  suggestionIndex = -1;
  $('command').removeAttribute('aria-activedescendant');
  for (const [index, text] of matches.entries()) {
    const button = document.createElement('button');
    button.type = 'button'; button.role = 'option'; button.textContent = text;
    button.id = `suggestion-${index}`; button.tabIndex = -1;
    button.setAttribute('aria-selected', 'false');
    button.onmousedown = event => event.preventDefault();
    button.onclick = () => applyText(text);
    $('suggestions').append(button);
  }
  if (!matches.length) {
    const empty = document.createElement('p');
    empty.textContent = 'No matching commands.';
    $('suggestions').append(empty);
  }
  $('suggestions').hidden = false;
  $('command').setAttribute('aria-expanded', 'true');
  $('browse-commands').setAttribute('aria-expanded', 'true');
}
$('command').addEventListener('input', () => showSuggestions());
$('command').addEventListener('focus', () => { if (connected) { $('command').select(); showSuggestions(''); } });
$('browse-commands').onclick = () => {
  const wasOpen = !$('suggestions').hidden;
  if (wasOpen) hideSuggestions();
  else { $('command').focus(); showSuggestions(''); }
};
$('browse-commands').onmousedown = event => event.preventDefault();
$('command').addEventListener('blur', hideSuggestions);
$('command').addEventListener('keydown', event => {
  if (event.key === 'Escape') { hideSuggestions(); return; }
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault();
    if ($('suggestions').hidden) showSuggestions();
    const options = [...$('suggestions').querySelectorAll('button')];
    if (!options.length) return;
    suggestionIndex = suggestionIndex < 0 ? (event.key === 'ArrowDown' ? 0 : options.length - 1) : (suggestionIndex + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length;
    options.forEach((option, index) => option.setAttribute('aria-selected', String(index === suggestionIndex)));
    $('command').setAttribute('aria-activedescendant', options[suggestionIndex].id);
    options[suggestionIndex].scrollIntoView({block: 'nearest'});
  } else if (event.key === 'Enter' && !$('suggestions').hidden && suggestionIndex >= 0) {
    event.preventDefault();
    applyText($('suggestions').querySelectorAll('button')[suggestionIndex].textContent);
  }
});
document.addEventListener('click', event => { if (!event.target.closest('#suggestions, #text-form, #browse-commands')) hideSuggestions(); });
$('text-form').onsubmit = event => { event.preventDefault(); applyText($('command').value.trim()); };
$('run').onclick = () => send(state?.running ? 'pause' : 'run');
$('reset').onclick = () => send('reset');
$('clear-goal').onclick = () => send('clear_goal');
document.querySelectorAll('[data-mode]').forEach(button => {
  button.onclick = () => { if (button.dataset.mode !== state?.mode) send('mode', {mode: button.dataset.mode}); };
});
$('goal-form').onsubmit = event => {
  event.preventDefault();
  send('goal', {xy: [Number($('goal-x').value), Number($('goal-y').value)]});
};
function renderState(changedMode) {
  const goal = state.mode === 'goal';
  $('goal-controls').hidden = !goal;
  document.querySelectorAll('[data-mode]').forEach(button => button.setAttribute('aria-selected', String(button.dataset.mode === state.mode)));
  if (changedMode) {
    $('mode-copy').replaceChildren();
    const h = document.createElement('h2'), p = document.createElement('p');
    h.textContent = goal ? 'Give motion a destination.' : 'A motion for every word.';
    p.textContent = goal ? 'Choose a motion style and place a goal. The policy steers using the true distance and direction.' : 'Select a motion or search the available text commands. Changes apply while the policy runs.';
    $('mode-copy').append(h, p);
    hideSuggestions();
    mapView = {cx: 0, cy: 0, span: 8};
  }
  $('text-hint').textContent = goal ? 'Five goal commands. Non-idle motions set the style for this goal and the next. Stand pauses the motion; Pause freezes physics.' : 'Try a quick pick, or browse and search all 130 training commands. Use ↑ / ↓ and Enter to select.';
  $('mode-badge').textContent = goal ? 'Goal reaching · ground truth' : 'Text-driven control';
  $('run').textContent = state.running ? 'Ⅱ Pause simulation' : '▶ Run simulation';
  $('status').textContent = state.error ? 'Reset needed' : state.running ? 'Running' : 'Paused';
  const minutes = Math.floor(state.time / 60), seconds = (state.time % 60).toFixed(2).padStart(5, '0');
  $('clock').textContent = `${String(minutes).padStart(2, '0')}:${seconds}`;
  if ($('motion').textContent !== state.text) $('motion').textContent = state.text;
  $('inference').replaceChildren(document.createTextNode(state.inference_ms.toFixed(1)), Object.assign(document.createElement('small'), {textContent: ' ms'}));
  $('speed').replaceChildren(document.createTextNode(state.speed.toFixed(2)), Object.assign(document.createElement('small'), {textContent: ' ×'}));
  $('position').replaceChildren(document.createTextNode(state.position.slice(0, 2).map(v => v.toFixed(2)).join(', ')), Object.assign(document.createElement('small'), {textContent: ' m'}));
  $('distance').replaceChildren(document.createTextNode(state.distance === null ? '—' : state.distance.toFixed(2)), Object.assign(document.createElement('small'), {textContent: ' m'}));
  $('goal-status').textContent = state.reached ? 'Goal reached' : state.goal ? (state.running ? 'Approaching' : 'Goal ready') : 'No goal';
  $('goal-body').textContent = `Body offset: ${state.goal_body ? state.goal_body.map(v => v.toFixed(2)).join(', ') + ' m' : '—'}`;
  $('threshold').textContent = state.threshold.toFixed(1);
  if (document.activeElement !== $('command') && !$('suggestions').contains(document.activeElement)) $('command').value = goal ? state.active_text : state.text;
  if (state.error) message(state.error, true);
  updatePresets();
  if (goal) drawMap();
}
function drawMap() {
  if (!state) return;
  const canvas = $('map'), rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const dpr = devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  mapView.cx = state.position[0]; mapView.cy = state.position[1];
  // Keep the robot and goal visible; equal units along both axes.
  mapView.span = state.goal ? Math.max(8, Math.abs(state.goal[0] - mapView.cx) * 2.5, Math.abs(state.goal[1] - mapView.cy) * 2.5 * w / h) : 8;
  const scale = w / mapView.span;
  const project = p => [w / 2 + (p[0] - mapView.cx) * scale, h / 2 - (p[1] - mapView.cy) * scale];
  ctx.strokeStyle = '#263222'; ctx.lineWidth = 1;
  const spacing = mapView.span > 20 ? 5 : 1;
  for (let x = Math.floor(mapView.cx - mapView.span / 2); x <= mapView.cx + mapView.span / 2; x++) {
    if (x % spacing) continue;
    const [px] = project([x, 0]); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
  }
  for (let y = Math.floor(mapView.cy - h / scale / 2); y <= mapView.cy + h / scale / 2; y++) {
    if (y % spacing) continue;
    const [, py] = project([0, y]); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(w, py); ctx.stroke();
  }
  ctx.strokeStyle = '#718b49'; ctx.lineWidth = 1.5; ctx.beginPath();
  state.trail.forEach((p, i) => { const xy = project(p); if (i) ctx.lineTo(...xy); else ctx.moveTo(...xy); }); ctx.stroke();
  if (state.goal) {
    const [gx, gy] = project(state.goal);
    ctx.fillStyle = '#edbc7520'; ctx.strokeStyle = state.reached ? '#9ddb80' : '#edbc75';
    ctx.beginPath(); ctx.arc(gx, gy, state.threshold * scale, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gx - 6, gy); ctx.lineTo(gx + 6, gy); ctx.moveTo(gx, gy - 6); ctx.lineTo(gx, gy + 6); ctx.stroke();
    ctx.fillStyle = '#edbc75'; ctx.font = '9px sans-serif'; ctx.fillText('GOAL', gx + 10, gy - 10);
    ctx.setLineDash([4, 5]); ctx.beginPath(); ctx.moveTo(w / 2, h / 2); ctx.lineTo(gx, gy); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.save(); ctx.translate(w / 2, h / 2); ctx.rotate(-state.yaw);
  ctx.fillStyle = '#d2f37b'; ctx.beginPath(); ctx.moveTo(10, 0); ctx.lineTo(-6, -6); ctx.lineTo(-3, 0); ctx.lineTo(-6, 6); ctx.closePath(); ctx.fill(); ctx.restore();
  ctx.fillStyle = '#89947f'; ctx.font = '9px sans-serif'; ctx.fillText('+Y ↑', 10, 16); ctx.fillText('+X →', w - 39, h - 11); ctx.fillText(`Grid ${spacing} m`, 10, h - 11);
}
$('map').onclick = event => {
  if (!state) return;
  const rect = $('map').getBoundingClientRect(), scale = rect.width / mapView.span;
  const xy = [mapView.cx + (event.clientX - rect.left - rect.width / 2) / scale, mapView.cy - (event.clientY - rect.top - rect.height / 2) / scale];
  $('goal-x').value = xy[0].toFixed(2); $('goal-y').value = xy[1].toFixed(2);
  send('goal', {xy});
};
let cameraTimer;
function sendCamera() { clearTimeout(cameraTimer); cameraTimer = setTimeout(() => send('camera', camera), 40); }
$('frame').onpointerdown = event => { dragging = [event.clientX, event.clientY]; $('frame').setPointerCapture(event.pointerId); event.preventDefault(); };
$('frame').onpointermove = event => {
  if (!dragging) return;
  camera.azimuth += (event.clientX - dragging[0]) * .4;
  camera.elevation = Math.max(-85, Math.min(-5, camera.elevation - (event.clientY - dragging[1]) * .25));
  dragging = [event.clientX, event.clientY]; sendCamera();
};
$('frame').onpointerup = $('frame').onpointercancel = () => { dragging = null; };
$('frame').addEventListener('wheel', event => { event.preventDefault(); camera.distance = Math.max(1.5, Math.min(15, camera.distance + event.deltaY * .005)); sendCamera(); }, {passive: false});
window.addEventListener('resize', drawMap);
setConnected(false);
connect();
