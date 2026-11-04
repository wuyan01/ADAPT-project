import {View} from './view.js';

let view, simulator;
try {
  document.getElementById('frame').hidden = true;
  view = new View(document.querySelector('.stage'));
} catch (e) {
  document.getElementById('loading').textContent = 'This demo needs WebGL graphics. Try a current desktop browser with hardware acceleration enabled.';
  throw e;
}

window.createLocalSimulator = () => {
  if (simulator) return simulator;
  const worker = new Worker(new URL('./worker.js', import.meta.url), {type: 'module'});
  simulator = {
    send(payload) { worker.postMessage(JSON.parse(payload)); },
    close() { worker.terminate(); simulator = null; },
    onmessage: null, onerror: null, onclose: null,
  };
  worker.onmessage = event => {
    const packet = event.data;
    if (packet.type === 'geometry') { view.geometry(packet.geoms); return; }
    if (packet.type === 'poses') { view.poses(packet); return; }
    if (packet.type === 'fatal') {
      document.getElementById('loading').hidden = false;
      document.getElementById('loading').textContent = packet.message;
      document.getElementById('connection').textContent = 'Unable to start';
      packet.type = 'error';
    }
    simulator.onmessage?.({data: JSON.stringify(packet)});
    if (packet.type === 'hello') {
      document.getElementById('connection').textContent = `On your device · ${packet.backend === 'webgpu' ? 'GPU (WebGPU)' : packet.gpuUnavailable ? 'CPU · WebGPU unavailable' : 'CPU'}`;
      if (packet.gpuUnavailable) document.getElementById('feedback').textContent = 'WebGPU is unavailable in this browser. Running on CPU; try Chrome or Edge with hardware acceleration enabled for GPU inference.';
    }
  };
  worker.onerror = event => {
    const text = 'The browser could not start the local runtime. Please try an up-to-date Chrome or Edge browser.';
    document.getElementById('loading').textContent = text;
    simulator.onmessage?.({data: JSON.stringify({type: 'error', message: text})});
    console.error(event.message);
  };
  return simulator;
};
document.getElementById('compute').addEventListener('change', event => simulator?.send(JSON.stringify({action: 'provider', provider: event.target.value})));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) simulator?.send(JSON.stringify({action: 'pause'}));
});
await import('./app.js');
