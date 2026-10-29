// Reads a recording file (.simrec, format in back/recording.py): a gzip stream of magic + header JSON + mesh arrays +
// frames. Used by playback (from the server or a local file) and the simple view's games.

const MAGIC = 'SIMREC01';

// compressed bytes (ArrayBuffer) -> {scene, meshes: {name: {vertices: Float32Array, indices: Uint32Array}},
// secondsPerFrame, frames (count), stride (floats per frame), data: Float32Array frames * stride (per frame time, then
// px, py, pz, qw, qx, qy, qz per body), info}
export async function parseSimrec(compressed) {
  const buffer = await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  if (magic !== MAGIC) throw new Error(`not a recording file (magic "${magic}", expected "${MAGIC}")`);
  const headerBytes = new DataView(buffer).getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 12, headerBytes)));
  let offset = 12 + headerBytes; // stays 4-byte aligned: the header is padded to a multiple of 4
  const meshes = {}; // {name: {vertices, indices}}
  for (const m of header.meshes) {
    const vertices = new Float32Array(buffer, offset, m.vertices * 3);
    offset += m.vertices * 12;
    const indices = new Uint32Array(buffer, offset, m.triangles * 3);
    offset += m.triangles * 12;
    meshes[m.name] = { vertices, indices };
  }
  const stride = 1 + 7 * header.scene.bodies.length;
  const data = new Float32Array(buffer, offset, header.frames * stride);
  if (offset + data.byteLength !== buffer.byteLength) throw new Error(`recording has ${buffer.byteLength - offset - data.byteLength} unexpected trailing bytes`);
  return { scene: header.scene, meshes, secondsPerFrame: header.seconds_per_frame, frames: header.frames, stride, data, info: header.info };
}
