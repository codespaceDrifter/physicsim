/**
 * API module for fetching recording data from backend.
 *
 * Binary format from backend (recording.py):
 * - Magic: 4 bytes "SIMB"
 * - secondsPerFrame: float32
 * - totalFrames: uint32
 * - frameCount: uint32
 * - For each frame:
 *   - time: float32
 *   - objectCount: uint32
 *   - For each object:
 *     - idLength: uint32, id: bytes
 *     - position: 3x float32
 *     - vertexCount: uint32, vertices: N x float32
 *     - indexCount: uint32, indices: N x uint32
 *     - colorLength: uint32, color: bytes
 */

export async function fetchRecording(worldName, chunkIndex) {
  const response = await fetch(`/recordings/${worldName}/${chunkIndex}`);
  if (!response.ok) {
    return null;
  }
  const buffer = await response.arrayBuffer();
  return decodeRecording(buffer);
}

function decodeRecording(buffer) {
  const view = new DataView(buffer);
  let offset = 0;

  // Skip magic header (4 bytes)
  offset += 4;

  const secondsPerFrame = view.getFloat32(offset, true);
  offset += 4;
  const totalFrames = view.getUint32(offset, true);
  offset += 4;
  const frameCount = view.getUint32(offset, true);
  offset += 4;

  const frames = {};
  for (let i = 0; i < frameCount; i++) {
    const time = view.getFloat32(offset, true);
    offset += 4;
    const objectCount = view.getUint32(offset, true);
    offset += 4;

    const idArray = [];
    const positionArrays = [];
    const vertexArrays = [];
    const indexArrays = [];
    const colorArray = [];

    for (let j = 0; j < objectCount; j++) {
      // ID (length-prefixed string)
      const idLength = view.getUint32(offset, true);
      offset += 4;
      const id = new TextDecoder().decode(new Uint8Array(buffer, offset, idLength));
      offset += idLength;
      idArray.push(id);

      // Position (3 floats)
      const position = [
        view.getFloat32(offset, true),
        view.getFloat32(offset + 4, true),
        view.getFloat32(offset + 8, true),
      ];
      offset += 12;
      positionArrays.push(position);

      // Vertices (count + floats)
      const vertexCount = view.getUint32(offset, true);
      offset += 4;
      const vertices = new Float32Array(buffer.slice(offset, offset + vertexCount * 4));
      offset += vertexCount * 4;
      vertexArrays.push(Array.from(vertices));

      // Indices (count + uints)
      const indexCount = view.getUint32(offset, true);
      offset += 4;
      const indices = new Uint32Array(buffer.slice(offset, offset + indexCount * 4));
      offset += indexCount * 4;
      indexArrays.push(Array.from(indices));

      // Color (length-prefixed string)
      const colorLength = view.getUint32(offset, true);
      offset += 4;
      const color = new TextDecoder().decode(new Uint8Array(buffer, offset, colorLength));
      offset += colorLength;
      colorArray.push(color);
    }

    const timeKey = time.toFixed(3);
    frames[timeKey] = {
      curTime: time,
      idArray,
      positionArrays,
      vertexArrays,
      indexArrays,
      colorArray,
    };
  }

  return {
    secondsPerFrame,
    totalFrames,
    frames,
  };
}
