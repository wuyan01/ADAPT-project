// Graph capture replays commands against the same GPU buffers on every step.
// MuJoCo still needs the actions on the CPU, so include readback in run().
export function createPolicyRunner(ort, session, inputs, graphCapture) {
  if (!graphCapture) return {
    async run(values) {
      const feeds = Object.fromEntries(inputs.map(x => [x.name, new ort.Tensor('float32', values[x.name], x.shape)]));
      const result = await session.run(feeds);
      try { return new Float32Array(await result[session.outputNames[0]].getData()); }
      finally { for (const tensor of Object.values(result)) tensor.dispose(); }
    },
    dispose() {},
  };

  const device = ort.env.webgpu.device, buffers = [], tensors = [];
  const allocate = (shape, usage) => {
    const size = shape.reduce((a, b) => a * b, 1) * 4;
    const buffer = device.createBuffer({size: Math.ceil(size / 16) * 16, usage});
    buffers.push(buffer);
    const tensor = ort.Tensor.fromGpuBuffer(buffer, {dataType: 'float32', dims: shape});
    tensors.push(tensor);
    return {buffer, tensor, size};
  };
  const dispose = () => {
    for (const tensor of tensors) tensor.dispose();
    for (const buffer of buffers) buffer.destroy();
  };
  try {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const feeds = {}, fetches = {};
    const uploads = inputs.map(x => {
      const slot = allocate(x.shape, usage); feeds[x.name] = slot.tensor;
      return {...x, ...slot};
    });
    const outputs = session.outputMetadata.map(x => {
      if (x.type !== 'float32' || !x.shape.length || x.shape.some(d => !Number.isInteger(d) || d <= 0)) {
        throw Error(`Graph capture needs a static float32 output: ${x.name}`);
      }
      const slot = allocate(x.shape, usage); fetches[x.name] = slot.tensor;
      return slot;
    });
    const action = outputs[0];
    if (action.size !== 29 * 4) throw Error('Policy must return 29 actions.');
    const readback = device.createBuffer({size: action.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    buffers.push(readback);
    return {
      async run(values) {
        for (const x of uploads) device.queue.writeBuffer(x.buffer, 0, values[x.name]);
        await session.run(feeds, fetches);
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(action.buffer, 0, readback, 0, action.size);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        try { return new Float32Array(readback.getMappedRange()).slice(); }
        finally { readback.unmap(); }
      },
      dispose,
    };
  } catch (e) { dispose(); throw e; }
}
