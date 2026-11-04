import loadMujoco from './vendor/mujoco/mujoco.js';
import * as ort from './vendor/onnx/ort.webgpu.bundle.min.mjs';

let mj, model, data, manifest, mode = 'text', requestedProvider = 'auto';
let running = false, error = null, text = 'stand', activeText = 'walk', goal = null;
let reached = false, insideSince = null, inferenceMs = 0, speed = 0, nextState = 0;
let history = {}, lastAction = new Float32Array(29), trail = [], seed = 42, fixedNoise;
let policy, ids, motorIds, qposIds, qvelIds, root, base, randomSpare;
const sessions = new Map(), commands = [];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const emit = value => postMessage(value);
const progress = message => emit({type: 'progress', message});
const json = async path => { const r = await fetch(new URL(path, import.meta.url)); if (!r.ok) throw Error(`Could not load ${path} (${r.status})`); return r.json(); };

function random() {
  seed |= 0; seed = seed + 0x6D2B79F5 | 0;
  let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
  return ((t ^ t >>> 14) >>> 0) / 4294967296;
}
function normal() {
  if (randomSpare !== undefined) { const v = randomSpare; randomSpare = undefined; return v; }
  const radius = Math.sqrt(-2 * Math.log(Math.max(1e-12, random()))), angle = 2 * Math.PI * random();
  randomSpare = radius * Math.sin(angle); return radius * Math.cos(angle);
}
function pose() {
  const p = Array.from(data.xpos.slice(root * 3, root * 3 + 3));
  const r = data.xmat.slice(root * 9, root * 9 + 9);
  return {p, r, yaw: Math.atan2(r[3], r[0])};
}
function bodyGoal(p, yaw) {
  if (!goal) return [0, 0];
  const dx = goal[0] - p[0], dy = goal[1] - p[1], c = Math.cos(yaw), s = Math.sin(yaw);
  return [c * dx + s * dy, -s * dx + c * dy];
}
function reset() {
  running = false; error = null; goal = null; reached = false; insideSince = null;
  text = 'stand'; activeText = policy.config.goal_monitor?.active_text_key || 'walk';
  history = {}; lastAction.fill(0); trail = []; seed = 42; fixedNoise = undefined; randomSpare = undefined;
  inferenceMs = 0; speed = 0;
  mj.mj_resetData(model, data);
  data.qpos.set(manifest.reset_qpos);
  for (let i = 0; i < 29; i++) data.qpos[qposIds[i]] = policy.config.default_joint_pos[i];
  mj.mj_forward(model, data);
}
function observation() {
  const cfg = policy.config, order = cfg.joint_ids_map, {p, r, yaw} = pose();
  const values = {
    goal_command: bodyGoal(p, yaw),
    base_lin_vel: cfg.zero_out_linvel ? [0, 0, 0] : Array.from(data.qvel.slice(0, 3)),
    base_ang_vel: Array.from(data.qvel.slice(3, 6)),
    projected_gravity: [-r[6], -r[7], -r[8]],
    joint_pos_rel: order.map(i => data.qpos[qposIds[i]] - cfg.default_joint_pos[i]),
    joint_vel_rel: order.map(i => data.qvel[qvelIds[i]]),
    last_action: lastAction,
    text_embedding: policy.embeddings[text],
  };
  const output = [];
  for (const [name, term] of Object.entries(cfg.observations)) {
    let value;
    if (name === 'noise') {
      const params = term.params, length = params.state_dim * (params.num_future_steps || 1), limit = params.clip_noise ?? 1;
      if (params.mode === 'fixed' && fixedNoise) value = fixedNoise.slice();
      else value = Array.from({length}, () => params.mode === 'zero' ? 0 : Math.max(-limit, Math.min(limit, normal())));
      if (params.mode === 'fixed') fixedNoise = value.slice();
    } else value = Array.from(values[name]);
    const scale = term.scale;
    for (let i = 0; i < value.length; i++) {
      if (term.clip) value[i] = Math.max(term.clip[0], Math.min(term.clip[1], value[i]));
      if (scale) value[i] *= scale[i] ?? (scale.every(s => s === scale[0]) ? scale[0] : 1);
    }
    const length = term.history_length || 1;
    if (!history[name]) history[name] = Array.from({length}, () => value.slice());
    history[name].push(value); history[name].shift();
    for (const previous of history[name]) output.push(...previous);
  }
  const expected = mode === 'text' ? 1088 : 994;
  if (output.length !== expected) throw Error(`Observation mismatch: ${output.length} / ${expected}`);
  return new Float32Array(output);
}
async function loadPolicy(nextMode) {
  running = false;
  const selected = manifest.policies[nextMode];
  const cacheKey = `${nextMode}:${requestedProvider}`;
  if (!sessions.has(cacheKey)) {
    const bytes = new Uint8Array(selected.bytes); let offset = 0;
    for (const part of selected.parts) {
      progress(`Loading ${nextMode === 'text' ? 'text' : 'goal'} policy: ${Math.round(offset / selected.bytes * 100)}% · ${(selected.bytes / 1024**2).toFixed(0)} MB on your device`);
      const response = await fetch(new URL(part, import.meta.url));
      if (!response.ok) throw Error(`Policy download failed (${response.status}). Please reload.`);
      const chunk = new Uint8Array(await response.arrayBuffer()); bytes.set(chunk, offset); offset += chunk.length;
    }
    if (offset !== bytes.length) throw Error('Incomplete policy download. Please reload.');
    progress('Preparing the policy on your device. This can take a moment…');
    let backend = 'wasm', session;
    if (requestedProvider !== 'wasm' && self.navigator.gpu) {
      try {
        session = await ort.InferenceSession.create(bytes, {executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all'});
        backend = 'webgpu';
      } catch (e) { progress('GPU initialization was unavailable. Loading the CPU version…'); }
    }
    if (!session) session = await ort.InferenceSession.create(bytes, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
    sessions.set(cacheKey, {session, backend});
  }
  mode = nextMode; policy = {...selected, ...sessions.get(cacheKey)}; reset();
  emit({type: 'hello', commands: Object.fromEntries(Object.entries(manifest.policies).map(([key, p]) => [key, Object.keys(p.embeddings)])), backend: policy.backend});
}
function state() {
  const {p, yaw} = pose();
  return {type: 'state', mode, running, text, active_text: activeText, time: data.time,
    position: p, yaw, goal, distance: goal ? Math.hypot(goal[0] - p[0], goal[1] - p[1]) : null,
    goal_body: goal ? bodyGoal(p, yaw) : null, reached, threshold: policy.config.goal_monitor?.reached_distance ?? .5,
    inference_ms: inferenceMs, speed: running ? speed : 0, error, trail: trail.filter((_, i) => i % 4 === 0), backend: policy.backend};
}
function sendState() {
  emit(state());
  emit({type: 'poses', positions: data.geom_xpos.slice(), rotations: data.geom_xmat.slice(), root: pose().p, goal, reached});
}
async function apply(cmd) {
  if (cmd.action === 'mode') {
    if (!manifest.policies[cmd.mode]) throw Error('Unknown mode.');
    await loadPolicy(cmd.mode);
  } else if (cmd.action === 'provider') {
    if (!['auto', 'wasm', 'webgpu'].includes(cmd.provider)) throw Error('Unknown compute device.');
    if (cmd.provider !== requestedProvider) {
      running = false;
      for (const previous of sessions.values()) await previous.session.release();
      sessions.clear();
    }
    requestedProvider = cmd.provider;
    await loadPolicy(mode);
  } else if (cmd.action === 'reset') reset();
  else if (cmd.action === 'pause') running = false;
  else if (cmd.action === 'run') {
    if (error) throw Error('Reset the robot before resuming.');
    running = true;
  } else if (cmd.action === 'text') {
    if (!Object.hasOwn(policy.embeddings, cmd.text)) throw Error('Choose a command from the allowed text set for this policy.');
    if (mode === 'goal' && cmd.text !== 'stand') { activeText = cmd.text; if (goal && !reached) text = cmd.text; }
    else text = cmd.text;
  } else if (cmd.action === 'goal') {
    if (mode !== 'goal') throw Error('Select Goal reaching first.');
    if (!Array.isArray(cmd.xy) || cmd.xy.length !== 2 || !cmd.xy.every(v => Number.isFinite(v) && Math.abs(v) <= 50)) throw Error('Goal coordinates must be within ±50 m.');
    goal = cmd.xy; reached = false; insideSince = null; text = activeText;
  } else if (cmd.action === 'clear_goal') { goal = null; reached = false; insideSince = null; text = 'stand'; }
  else if (cmd.action !== 'camera') throw Error('Unknown command.');
  emit({type: 'ack', action: cmd.action}); sendState();
}
async function advance() {
  const cfg = policy.config, monitor = cfg.goal_monitor || {}, start = performance.now();
  if (mode === 'goal' && goal && !reached) {
    const p = pose().p, distance = Math.hypot(goal[0] - p[0], goal[1] - p[1]);
    if (distance <= (monitor.reached_distance ?? .5)) {
      insideSince ??= data.time;
      if (data.time - insideSince >= (monitor.reached_hold_time ?? .3)) { reached = true; text = monitor.idle_text_key || 'stand'; }
    } else insideSince = null;
  }
  const obs = observation(), before = performance.now();
  // Physics deliberately waits for the asynchronous local inference result.
  const result = await policy.session.run({obs: new ort.Tensor('float32', obs, [1, obs.length])});
  inferenceMs = performance.now() - before;
  const actions = result[policy.session.outputNames[0]].data;
  if (actions.length !== 29 || !actions.every(Number.isFinite)) throw Error('Policy returned invalid actions.');
  const target = new Float64Array(29), actionCfg = cfg.actions.JointPositionAction;
  for (let i = 0; i < 29; i++) { lastAction[i] = actions[i]; target[cfg.joint_ids_map[i]] = actions[i]; }
  for (let i = 0; i < 29; i++) target[i] = target[i] * actionCfg.scale[i] + actionCfg.offset[i];
  for (let step = 0; step < Math.round(cfg.step_dt / .002); step++) {
    for (let i = 0; i < 29; i++) data.ctrl[motorIds[i]] = cfg.stiffness[i] * (target[i] - data.qpos[qposIds[i]]) - cfg.damping[i] * data.qvel[qvelIds[i]];
    mj.mj_step(model, data);
  }
  mj.mj_forward(model, data);
  if (!data.qpos.every(Number.isFinite) || data.qpos[2] < .35) throw Error('Robot fell. Reset to try another command.');
  trail.push(pose().p.slice(0, 2)); if (trail.length > 1200) trail.shift();
  const dt = performance.now() - start;
  return dt;
}
function geometry() {
  const geoms = [], cache = new Map();
  for (let i = 0; i < model.ngeom; i++) {
    if (!model.geom_bodyid[i] || model.geom_group[i] !== 1) continue;
    const meshId = model.geom_dataid[i], type = model.geom_type[i];
    const item = {id: i, type, color: Array.from(model.geom_rgba.slice(i * 4, i * 4 + 4)), size: Array.from(model.geom_size.slice(i * 3, i * 3 + 3))};
    if (type === 7) {
      if (!cache.has(meshId)) {
        const va = model.mesh_vertadr[meshId], vn = model.mesh_vertnum[meshId], fa = model.mesh_faceadr[meshId], fn = model.mesh_facenum[meshId];
        cache.set(meshId, {vertices: new Float32Array(model.mesh_vert.slice(va * 3, (va + vn) * 3)), faces: new Uint32Array(model.mesh_face.slice(fa * 3, (fa + fn) * 3))});
      }
      Object.assign(item, cache.get(meshId));
    }
    geoms.push(item);
  }
  emit({type: 'geometry', geoms});
}
async function init() {
  try {
    manifest = await json('./manifest.json');
    progress('Loading MuJoCo physics on your device…');
    mj = await loadMujoco();
    mj.FS.mkdirTree('/robot/meshes');
    await Promise.all(manifest.robot_files.map(async path => {
      const response = await fetch(new URL(path, import.meta.url));
      if (!response.ok) throw Error(`Could not load robot asset (${response.status}).`);
      mj.FS.writeFile('/robot/' + path.replace('assets/robot/', ''), new Uint8Array(await response.arrayBuffer()));
    }));
    model = mj.MjModel.from_xml_path('/robot/g1.xml');
    model.opt.timestep = .002;
    for (let i = 0; i < model.ngeom; i++) if (model.geom_contype[i] || model.geom_conaffinity[i]) {
      model.geom_contype[i] = model.geom_bodyid[i] ? 1 : 2;
      model.geom_conaffinity[i] = model.geom_bodyid[i] ? 2 : 1;
    }
    data = new mj.MjData(model);
    root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, 'pelvis');
    ids = manifest.joint_names.map(n => mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT.value, n));
    if (root < 0 || ids.some(i => i < 0)) throw Error('G1 joint mapping is incomplete.');
    qposIds = ids.map(i => model.jnt_qposadr[i]); qvelIds = ids.map(i => model.jnt_dofadr[i]);
    motorIds = ids.map(i => { for (let a = 0; a < model.nu; a++) if (model.actuator_trnid[a * 2] === i) return a; throw Error('Missing torque motor.'); });
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = new URL('./vendor/onnx/', import.meta.url).href;
    await loadPolicy('text'); geometry(); sendState();
    while (true) {
      const start = performance.now();
      while (commands.length) {
        try { await apply(commands.shift()); } catch (e) { emit({type: 'error', message: e.message}); }
      }
      if (running) {
        try { await advance(); } catch (e) { running = false; error = e.message; }
      }
      if (performance.now() >= nextState) { sendState(); nextState = performance.now() + 33; }
      const period = running ? policy.config.step_dt * 1000 : 33;
      await sleep(Math.max(0, period - (performance.now() - start)));
      speed = period / Math.max(period, performance.now() - start);
    }
  } catch (e) { emit({type: 'fatal', message: `Could not start the local simulation: ${e.message}`}); }
}
self.onmessage = event => { if (commands.length < 30) commands.push(event.data); };
init();
