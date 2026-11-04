import test from 'node:test';
import assert from 'node:assert/strict';
import {createPolicyRunner} from '../policy-runner.mjs';

test('capture reuses GPU bindings, uploads changed observations and reads fresh actions', async () => {
  globalThis.GPUBufferUsage = {STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, MAP_READ: 8};
  globalThis.GPUMapMode = {READ: 1};
  const buffers = [], bindings = [];
  const device = {
    createBuffer({size}) {
      const buffer = {bytes: new ArrayBuffer(size), destroyed: false,
        async mapAsync() {}, getMappedRange() { return this.bytes; }, unmap() {},
        destroy() { this.destroyed = true; }};
      buffers.push(buffer); return buffer;
    },
    queue: {
      writeBuffer(buffer, offset, values) { new Uint8Array(buffer.bytes, offset, values.byteLength).set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength)); },
      submit(commands) { for (const command of commands) command(); },
    },
    createCommandEncoder() {
      let copy;
      return {copyBufferToBuffer(source, from, target, to, size) {
        copy = () => new Uint8Array(target.bytes, to, size).set(new Uint8Array(source.bytes, from, size));
      }, finish() { return copy; }};
    },
  };
  const ort = {env: {webgpu: {device}}, Tensor: {fromGpuBuffer: (gpuBuffer) => ({gpuBuffer, dispose() {}})}};
  const session = {outputMetadata: [{name: 'actions', type: 'float32', shape: [1, 29]}],
    async run(feeds, fetches) {
      bindings.push([feeds.obs.gpuBuffer, fetches.actions.gpuBuffer]);
      new Float32Array(fetches.actions.gpuBuffer.bytes).fill(new Float32Array(feeds.obs.gpuBuffer.bytes)[0]);
    }};
  const runner = createPolicyRunner(ort, session, [{name: 'obs', shape: [1, 2]}], true);
  const first = await runner.run({obs: new Float32Array([1, 2])});
  const second = await runner.run({obs: new Float32Array([3, 4])});
  assert.deepEqual(Array.from(first), Array(29).fill(1));
  assert.deepEqual(Array.from(second), Array(29).fill(3));
  assert.equal(bindings[0][0], bindings[1][0]);
  assert.equal(bindings[0][1], bindings[1][1]);
  runner.dispose();
  assert.ok(buffers.every(b => b.destroyed));
});

test('ordinary sessions download GPU outputs and dispose results even on readback failure', async () => {
  let disposed = 0;
  const ort = {Tensor: class {constructor(type, data, shape) { Object.assign(this, {type, data, shape}); }}};
  const session = {outputNames: ['actions'], async run(feeds) {
    return {actions: {async getData() { return feeds.obs.data; }, dispose() { disposed++; }}};
  }};
  const runner = createPolicyRunner(ort, session, [{name: 'obs', shape: [1, 2]}], false);
  assert.deepEqual(Array.from(await runner.run({obs: new Float32Array([5, 6])})), [5, 6]);
  session.run = async () => ({actions: {async getData() { throw Error('download failed'); }, dispose() { disposed++; }}});
  await assert.rejects(runner.run({obs: new Float32Array(2)}), /download failed/);
  assert.equal(disposed, 2);
});
