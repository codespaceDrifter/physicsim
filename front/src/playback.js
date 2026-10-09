// Playback mode: stream recording chunks (binary format: spec section 3) and play them back on a timeline.
import { app, on, emit, select } from './state.js';
import { getJson, getBinary } from './api.js';
import { applyPoses, bodyGroups } from './sceneView.js';
import { onTick, frameObjects } from './viewport.js';

const recSelect = document.getElementById('rec-select');
const playBtn = document.getElementById('pb-play');
const speedSelect = document.getElementById('pb-speed');
const slider = document.getElementById('pb-slider');
const timeEl = document.getElementById('pb-time');
const statusTimeEl = document.getElementById('status-time');

const FRAMES_PER_CHUNK = 1000;
const HEADER_BYTES = 20; // magic, seconds_per_frame, total_frames, frames_in_chunk, num_bodies

const pb = {
  name: null, // recording name being played, null = none loaded
  secondsPerFrame: 0,
  totalFrames: 0,
  frame: 0, // fractional playhead in frames
  playing: false,
  // chunk index -> Float32Array of frames_in_chunk * (1 + 7 * num_bodies): per frame time then poses
  chunks: new Map(), // {chunkIndex: Float32Array}
  pending: new Set(), // {chunkIndex} requests in flight
};

async function fetchChunk(index) {
  if (pb.chunks.has(index) || pb.pending.has(index)) return;
  pb.pending.add(index);
  const name = pb.name;
  const buffer = await getBinary(`/recordings/${encodeURIComponent(name)}/${index}`);
  pb.pending.delete(index);
  // a different recording was loaded while this request was in flight
  if (name !== pb.name) return;
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== 'SIMR') throw new Error(`recording chunk ${index}: bad magic "${magic}"`);
  pb.secondsPerFrame = view.getFloat32(4, true);
  pb.totalFrames = view.getUint32(8, true);
  const framesInChunk = view.getUint32(12, true);
  const numBodies = view.getUint32(16, true);
  if (numBodies !== app.shown.bodies.length) {
    throw new Error(`recording chunk ${index} has ${numBodies} bodies, scene has ${app.shown.bodies.length}`);
  }
  const expected = Math.min(FRAMES_PER_CHUNK, pb.totalFrames - index * FRAMES_PER_CHUNK);
  if (framesInChunk !== expected) throw new Error(`recording chunk ${index} has ${framesInChunk} frames, expected ${expected}`);
  // header is 20 bytes so the float data is 4-byte aligned; little-endian like every browser platform
  pb.chunks.set(index, new Float32Array(buffer, HEADER_BYTES, framesInChunk * (1 + 7 * numBodies)));
  slider.max = pb.totalFrames - 1;
  // the playhead was waiting for this chunk (initial load / scrubbing)
  if (index === Math.floor(Math.floor(pb.frame) / FRAMES_PER_CHUNK)) showFrame();
}

// show the frame under the playhead; returns false if its chunk isn't loaded yet
function showFrame() {
  const frame = Math.floor(pb.frame);
  const chunkIndex = Math.floor(frame / FRAMES_PER_CHUNK);
  const data = pb.chunks.get(chunkIndex);
  if (!data) {
    fetchChunk(chunkIndex);
    return false;
  }
  const stride = 1 + 7 * app.shown.bodies.length; // floats per frame
  const offset = (frame - chunkIndex * FRAMES_PER_CHUNK) * stride;
  applyPoses(data.subarray(offset + 1, offset + stride));
  const duration = (pb.totalFrames - 1) * pb.secondsPerFrame; // time of the last frame
  timeEl.textContent = `${data[offset].toFixed(3)} / ${duration.toFixed(3)} s`;
  statusTimeEl.textContent = `t = ${data[offset].toFixed(3)} s · frame ${frame + 1}/${pb.totalFrames}`;
  slider.value = frame;

  // prefetch the next chunk, drop chunks far behind / ahead
  if ((chunkIndex + 1) * FRAMES_PER_CHUNK < pb.totalFrames) fetchChunk(chunkIndex + 1);
  for (const index of pb.chunks.keys()) {
    if (index < chunkIndex - 1 || index > chunkIndex + 2) pb.chunks.delete(index);
  }
  return true;
}

function setPlaying(playing) {
  pb.playing = playing;
  playBtn.textContent = playing ? '❚❚ Pause' : '▶ Play';
}

async function load(name) {
  setPlaying(false);
  pb.name = name;
  pb.chunks.clear();
  pb.pending.clear();
  pb.frame = 0;
  const scene = await getJson(`/recordings/${encodeURIComponent(name)}/scene`);
  // left playback mode / picked another recording while loading
  if (pb.name !== name) return;
  app.shown = scene;
  app.selection = [];
  emit('scene', 'playback');
  frameObjects(bodyGroups.filter(g => !g.userData.isPlane));
  await fetchChunk(0);
}

onTick((dt) => {
  if (app.mode !== 'playback' || !pb.playing) return;
  const prev = pb.frame;
  pb.frame = Math.min(pb.frame + (dt * Number(speedSelect.value)) / pb.secondsPerFrame, pb.totalFrames - 1);
  // hold the playhead while its chunk is still downloading
  if (!showFrame()) pb.frame = prev;
  if (pb.frame === pb.totalFrames - 1) setPlaying(false);
});

recSelect.onchange = () => load(recSelect.value);

playBtn.onclick = () => {
  if (pb.name === null) return;
  // play again from the start once at the end
  if (!pb.playing && Math.floor(pb.frame) === pb.totalFrames - 1) pb.frame = 0;
  setPlaying(!pb.playing);
};

slider.oninput = () => {
  if (pb.name === null) return;
  pb.frame = Number(slider.value);
  showFrame();
};

on('mode', async (prev) => {
  if (prev === 'playback') {
    setPlaying(false);
    pb.name = null;
    pb.chunks.clear();
    statusTimeEl.textContent = '';
  }
  if (app.mode !== 'playback') return;
  // entering playback: list recordings and open the first one
  app.shown = null;
  select([]);
  emit('scene', 'playback');
  const { recordings } = await getJson('/recordings');
  recSelect.innerHTML = '';
  for (const name of recordings) recSelect.add(new Option(name, name));
  if (recordings.length > 0) await load(recordings[0]);
});
