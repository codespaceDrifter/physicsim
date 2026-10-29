// Playback mode: view a recording (.simrec, see simrec.js) on a timeline with the normal camera controls. recordings
// come from the server's list (logs/), from "Open…", or from a file dropped anywhere on the page.
import { app, on, emit, select, setMode, setView } from './state.js';
import { getJson, getBinary } from './api.js';
import { applyPoses, bodyGroups } from './sceneView.js';
import { onTick, frameObjects } from './viewport.js';
import { parseSimrec } from './simrec.js';
import { toast } from './toast.js';

const recSelect = document.getElementById('rec-select');
const openBtn = document.getElementById('pb-open');
const fileInput = document.getElementById('pb-file');
const playBtn = document.getElementById('pb-play');
const speedSelect = document.getElementById('pb-speed');
const slider = document.getElementById('pb-slider');
const timeEl = document.getElementById('pb-time');
const statusTimeEl = document.getElementById('status-time');
const captionEl = document.getElementById('pb-caption');

const pb = {
  rec: null, // the loaded recording (parseSimrec), null = none
  frame: 0, // fractional playhead in frames
  playing: false,
  file: null, // a local file waiting for playback mode to open it
};

function showFrame() {
  const rec = pb.rec;
  const frame = Math.floor(pb.frame);
  const offset = frame * rec.stride;
  applyPoses(rec.data.subarray(offset + 1, offset + rec.stride));
  const time = rec.data[offset];
  const duration = rec.data[(rec.frames - 1) * rec.stride]; // time of the last frame
  timeEl.textContent = `${time.toFixed(3)} / ${duration.toFixed(3)} s`;
  statusTimeEl.textContent = `t = ${time.toFixed(3)} s · frame ${frame + 1}/${rec.frames}`;
  slider.value = frame;
  // a demo recording: its request and the plan line running at this time (the last caption started by now)
  const captions = rec.info.captions;
  captionEl.hidden = !captions;
  if (captions) {
    const now = captions.filter(([start]) => start <= time).pop();
    captionEl.innerHTML = '';
    captionEl.append(`"${rec.info.request}"`, document.createElement('br'));
    const line = document.createElement('b');
    line.textContent = now ? now[1] : '';
    captionEl.append(line);
  }
}

function setPlaying(playing) {
  pb.playing = playing;
  playBtn.textContent = playing ? '❚❚ Pause' : '▶ Play';
}

// show a recording: compressed .simrec bytes (ArrayBuffer), name for the status line
async function show(bytes, name) {
  setPlaying(false);
  const rec = await parseSimrec(bytes);
  // left playback mode while loading
  if (app.mode !== 'playback') return;
  pb.rec = rec;
  pb.frame = 0;
  slider.max = rec.frames - 1;
  // merged: mesh names are paths under robot/assets, so a name always means the same mesh
  app.meshes = { ...app.meshes, ...rec.meshes };
  app.shown = { ...rec.scene, name };
  // the list shows which server recording this is (blank for a local file)
  recSelect.value = [...recSelect.options].some(o => o.value === name) ? name : '';
  app.selection = [];
  emit('scene', 'playback');
  frameObjects(bodyGroups.filter(g => !g.userData.isPlane));
  showFrame();
  // what the recorder said about the run (e.g. hits, falls), in the console for whoever's curious
  if (Object.keys(rec.info).length) console.log(`${name}:`, rec.info);
}

async function openFile(file) {
  try {
    await show(await file.arrayBuffer(), file.name.replace(/\.simrec$/, ''));
  } catch (err) {
    toast(`${file.name}: ${err.message}`);
  }
}

onTick((dt) => {
  if (app.mode !== 'playback' || !pb.playing) return;
  const rec = pb.rec;
  pb.frame = Math.min(pb.frame + (dt * Number(speedSelect.value)) / rec.secondsPerFrame, rec.frames - 1);
  showFrame();
  if (pb.frame === rec.frames - 1) setPlaying(false);
});

recSelect.onchange = async () => show(await getBinary(`/recordings/${encodeURIComponent(recSelect.value)}`), recSelect.value);
openBtn.onclick = () => fileInput.click();
fileInput.onchange = () => {
  if (fileInput.files.length) openFile(fileInput.files[0]);
  fileInput.value = '';
};

// drop a .simrec anywhere: switch to the advanced view's playback mode and open it
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (app.mode === 'playback') {
    openFile(file);
    return;
  }
  pb.file = file;
  setView('advanced');
  setMode('playback');
});

playBtn.onclick = () => {
  if (pb.rec === null) return;
  // play again from the start once at the end
  if (!pb.playing && Math.floor(pb.frame) === pb.rec.frames - 1) pb.frame = 0;
  setPlaying(!pb.playing);
};

slider.oninput = () => {
  if (pb.rec === null) return;
  pb.frame = Number(slider.value);
  showFrame();
};

on('mode', async (prev) => {
  if (prev === 'playback') {
    setPlaying(false);
    pb.rec = null;
    statusTimeEl.textContent = '';
  }
  if (app.mode !== 'playback') return;
  // entering playback: list the server's recordings, open the dropped file or else the first recording
  app.shown = null;
  select([]);
  emit('scene', 'playback');
  const { recordings } = await getJson('/recordings');
  recSelect.innerHTML = '';
  for (const name of recordings) recSelect.add(new Option(name, name));
  if (pb.file) {
    const file = pb.file;
    pb.file = null;
    await openFile(file);
  } else if (recordings.length > 0) {
    await show(await getBinary(`/recordings/${encodeURIComponent(recordings[0])}`), recordings[0]);
  }
});
