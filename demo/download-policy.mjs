// Stream each part straight into the final allocation instead of retaining a
// second, up-to-48-MiB ArrayBuffer while copying it into the assembled model.
export async function downloadPolicy(policy, baseUrl, onProgress, fetcher = fetch) {
  const bytes = new Uint8Array(policy.bytes);
  let offset = 0;
  const append = chunk => {
    if (offset + chunk.byteLength > bytes.length) throw Error('Policy download has an unexpected size. Please reload.');
    bytes.set(chunk, offset); offset += chunk.byteLength;
    onProgress(offset, bytes.length);
  };
  onProgress(0, bytes.length);
  for (const part of policy.parts) {
    const response = await fetcher(new URL(part, baseUrl));
    if (!response.ok) throw Error(`Policy download failed (${response.status}). Please reload.`);
    if (!response.body?.getReader) { append(new Uint8Array(await response.arrayBuffer())); continue; }
    const reader = response.body.getReader();
    try {
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        append(value);
      }
    } catch (e) { await reader.cancel().catch(() => {}); throw e; }
    finally { reader.releaseLock(); }
  }
  if (offset !== bytes.length) throw Error('Incomplete policy download. Please reload.');
  return bytes;
}
