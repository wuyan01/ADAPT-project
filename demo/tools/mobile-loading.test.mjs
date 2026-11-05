import test from 'node:test';
import assert from 'node:assert/strict';
import {isMobileDevice} from '../runtime-profile.mjs';
import {downloadPolicy} from '../download-policy.mjs';

test('phone/tablet detection includes iPad desktop user agents and preserves desktops', () => {
  assert.ok(isMobileDevice({userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)'}));
  assert.ok(isMobileDevice({userAgent: 'Mozilla/5.0 (Linux; Android 15)'}));
  assert.ok(isMobileDevice({platform: 'MacIntel', maxTouchPoints: 5}));
  assert.ok(isMobileDevice({userAgentData: {mobile: true}}));
  assert.equal(isMobileDevice({platform: 'MacIntel', maxTouchPoints: 0}), false);
  assert.equal(isMobileDevice({userAgent: 'Mozilla/5.0 (Windows NT 10.0)', maxTouchPoints: 10}), false);
});

test('model parts stream in order without materializing a second part-sized buffer', async () => {
  const urls = [], progress = [];
  const parts = [[new Uint8Array([1, 2]), new Uint8Array([3])], [new Uint8Array([4, 5])]];
  const fetcher = async url => {
    urls.push(url.href);
    const chunks = parts.shift();
    return {ok: true, body: new ReadableStream({start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    }}), arrayBuffer() { throw Error('must stream'); }};
  };
  const bytes = await downloadPolicy({bytes: 5, parts: ['assets/a', 'assets/b']}, 'https://example.com/demo/worker.js', loaded => progress.push(loaded), fetcher);
  assert.deepEqual([...bytes], [1, 2, 3, 4, 5]);
  assert.deepEqual(urls, ['https://example.com/demo/assets/a', 'https://example.com/demo/assets/b']);
  assert.deepEqual(progress, [0, 2, 3, 5]);
});

test('incomplete, oversized and failed downloads cannot become a model', async () => {
  const policy = {bytes: 2, parts: ['assets/a']};
  const download = response => downloadPolicy(policy, 'https://example.com/demo/', () => {}, async () => response);
  await assert.rejects(download(new Response(new Uint8Array([1]))), /Incomplete/);
  await assert.rejects(download(new Response(new Uint8Array([1, 2, 3]))), /unexpected size/);
  await assert.rejects(download(new Response(null, {status: 404})), /404/);
});

test('older fetch implementations can load a part without a readable stream', async () => {
  const result = await downloadPolicy({bytes: 2, parts: ['model']}, 'https://example.com/demo/', () => {}, async () => ({
    ok: true, async arrayBuffer() { return new Uint8Array([8, 9]).buffer; },
  }));
  assert.deepEqual([...result], [8, 9]);
});
