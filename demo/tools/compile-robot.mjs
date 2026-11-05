// Build with the exact WASM runtime shipped to browsers; MJB is version-specific.
// Run from the repository root: node demo/tools/compile-robot.mjs
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import loadMujoco from '../vendor/mujoco/mujoco.js';

const directory = fileURLToPath(new URL('../', import.meta.url));
const manifestPath = path.join(directory, 'manifest.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath));
const mj = await loadMujoco();
mj.FS.mkdirTree('/robot/meshes');
for (const file of manifest.robot_files) {
  mj.FS.writeFile('/robot/' + file.replace('assets/robot/', ''), fs.readFileSync(path.join(directory, file)));
}
const source = mj.MjModel.from_xml_path('/robot/g1.xml');
const dataA = new mj.MjData(source);
const xmlHeapMiB = dataA.qpos.buffer.byteLength / 1024**2;
mj.mj_saveModel(source, '/g1.mjb', null);
const binary = mj.FS.readFile('/g1.mjb');

// Reload into a fresh runtime to measure its heap without the XML compiler's
// high-water mark, and compare geometry and contact simulation to the source.
const targetRuntime = await loadMujoco();
targetRuntime.FS.writeFile('/g1.mjb', binary, {canOwn: true});
const vfs = new targetRuntime.MjVFS();
const target = targetRuntime.MjModel.from_binary_path('/g1.mjb', vfs);
vfs.delete(); targetRuntime.FS.unlink('/g1.mjb');
for (const key of ['nq', 'nv', 'nu', 'ngeom', 'nbody', 'nmeshvert', 'nmeshface']) assert.equal(target[key], source[key], key);
for (const key of ['body_mass', 'body_inertia', 'mesh_vert', 'mesh_face', 'mesh_pos', 'mesh_quat', 'geom_pos', 'geom_quat', 'geom_size']) {
  assert.deepEqual(target[key], source[key], key);
}
const dataB = new targetRuntime.MjData(target);
for (const [runtime, model, data] of [[mj, source, dataA], [targetRuntime, target, dataB]]) {
  model.opt.timestep = .002;
  for (let i = 0; i < model.ngeom; i++) if (model.geom_contype[i] || model.geom_conaffinity[i]) {
    model.geom_contype[i] = model.geom_bodyid[i] ? 1 : 2;
    model.geom_conaffinity[i] = model.geom_bodyid[i] ? 2 : 1;
  }
  data.qpos.set(manifest.reset_qpos);
  runtime.mj_forward(model, data);
}
for (let step = 0; step < 250; step++) {
  for (let i = 0; i < source.nu; i++) dataA.ctrl[i] = dataB.ctrl[i] = .1 * Math.sin(step * .02 + i);
  mj.mj_step(source, dataA); targetRuntime.mj_step(target, dataB);
  assert.ok(dataA.qpos.every(Number.isFinite), `finite state at step ${step}`);
  assert.deepEqual(dataB.qpos, dataA.qpos, `qpos at step ${step}`);
  assert.deepEqual(dataB.qvel, dataA.qvel, `qvel at step ${step}`);
}

const digest = createHash('sha256').update(binary).digest('hex');
const compressed = gzipSync(binary, {level: 9});
const asset = `assets/robot/g1-${digest.slice(0, 12)}.mjb.gz`;
fs.writeFileSync(path.join(directory, asset), compressed);
manifest.robot_binary = {
  parts: [asset], bytes: binary.byteLength, compressed_bytes: compressed.byteLength,
  sha256: digest, wasm_sha256: createHash('sha256').update(fs.readFileSync(path.join(directory, 'vendor/mujoco/mujoco.wasm'))).digest('hex'),
};
fs.writeFileSync(manifestPath, JSON.stringify(manifest));
console.log(JSON.stringify({asset, compressedBytes: compressed.byteLength,
  xmlHeapMiB,
  binaryHeapMiB: dataB.qpos.buffer.byteLength / 1024**2,
  verification: 'geometry and 250 contact simulation steps match exactly'}, null, 2));
