// Top toolbar + viewport view buttons: scene load/new/save, edit actions, mode switch, sim controls.
import { app, on, emit, setMode } from './state.js';
import { getJson, putJson, getBinary } from './api.js';
import { newScene } from './model.js';
import { toast } from './toast.js';
import { setStandardView, setOrthographic } from './viewport.js';
import { addPrimitive, insertLekiwi, createJoint, duplicateSelection, deleteSelection, setGizmoMode, toggleSnap, frameSelection } from './editor.js';
import { sim, send, play } from './simClient.js';

const sceneSelect = document.getElementById('scene-select');

// buttons never keep keyboard focus, so Space / keys always go to teleop + shortcuts
document.addEventListener('mousedown', (e) => {
  if (e.target.closest('button')) e.preventDefault();
});

// ---------- scenes ----------
export async function refreshSceneList() {
  const { scenes } = await getJson('/scenes');
  sceneSelect.innerHTML = '';
  for (const name of scenes) sceneSelect.add(new Option(name, name));
  // an unsaved new scene isn't on the server yet
  if (app.scene && !scenes.includes(app.scene.name)) sceneSelect.add(new Option(`${app.scene.name} (unsaved)`, app.scene.name));
  if (app.scene) sceneSelect.value = app.scene.name;
  return scenes;
}

export function setScene(scene) {
  app.scene = scene;
  app.shown = scene;
  app.selection = [];
  emit('scene', 'load');
  frameSelection();
}

export async function loadScene(name) {
  const scene = await getJson(`/scenes/${encodeURIComponent(name)}`);
  // the visual meshes its bodies show (imported robots, HumanoidBench scenes), fetched once: /meshes answers
  // uint32 V, uint32 T, V x 3 float32 vertices, T x 3 uint32 triangles
  const names = new Set(scene.bodies.flatMap(b => (b.visuals ?? []).map(v => v.mesh)));
  await Promise.all([...names].filter(n => !(n in app.meshes)).map(async (n) => {
    const buffer = await getBinary(`/meshes/${n}`);
    const [v, t] = new Uint32Array(buffer, 0, 2);
    app.meshes[n] = { vertices: new Float32Array(buffer, 8, v * 3), indices: new Uint32Array(buffer, 8 + v * 12, t * 3) };
  }));
  setScene(scene);
  sceneSelect.value = name;
}

sceneSelect.onchange = () => loadScene(sceneSelect.value);

document.getElementById('btn-new').onclick = () => {
  const name = prompt('New scene name', 'untitled');
  if (name === null) return;
  if (!name.trim()) {
    toast('scene name cannot be empty');
    return;
  }
  setScene(newScene(name.trim()));
  sceneSelect.add(new Option(`${app.scene.name} (unsaved)`, app.scene.name));
  sceneSelect.value = app.scene.name;
};

document.getElementById('btn-save').onclick = async () => {
  if (!app.scene.name) {
    toast('scene needs a name (scene settings) before saving');
    return;
  }
  await putJson(`/scenes/${encodeURIComponent(app.scene.name)}`, app.scene);
  toast(`saved scene "${app.scene.name}"`, 'info');
  await refreshSceneList();
};

// ---------- edit actions ----------
document.getElementById('btn-box').onclick = () => addPrimitive('box');
document.getElementById('btn-sphere').onclick = () => addPrimitive('sphere');
document.getElementById('btn-cylinder').onclick = () => addPrimitive('cylinder');
document.getElementById('btn-lekiwi').onclick = () => insertLekiwi();
document.getElementById('btn-joint').onclick = () => createJoint(document.getElementById('joint-type').value);
document.getElementById('btn-duplicate').onclick = () => duplicateSelection();
document.getElementById('btn-delete').onclick = () => deleteSelection();
document.getElementById('btn-translate').onclick = () => setGizmoMode('translate');
document.getElementById('btn-rotate').onclick = () => setGizmoMode('rotate');
document.getElementById('btn-snap').onclick = () => toggleSnap();

// ---------- view ----------
for (const btn of document.querySelectorAll('#viewtools [data-view]')) {
  btn.onclick = () => setStandardView(btn.dataset.view);
}
let ortho = false;
document.getElementById('btn-ortho').onclick = () => {
  ortho = !ortho;
  setOrthographic(ortho);
  document.getElementById('btn-ortho').textContent = ortho ? 'Ortho' : 'Persp';
};
document.getElementById('btn-fit').onclick = () => frameSelection();

// ---------- modes ----------
const modeButtons = document.querySelectorAll('.modes button');
for (const btn of modeButtons) btn.onclick = () => setMode(btn.dataset.mode);
on('mode', () => {
  for (const btn of modeButtons) btn.classList.toggle('on', btn.dataset.mode === app.mode);
});

// ---------- simulation ----------
const playBtn = document.getElementById('btn-play');
const pauseBtn = document.getElementById('btn-pause');
const stepBtn = document.getElementById('btn-step');
playBtn.onclick = () => play();
pauseBtn.onclick = () => send({ type: sim.state === 'paused' ? 'resume' : 'pause' });
stepBtn.onclick = () => send({ type: 'step' });
// stop = discard the world and go back to editing the unmodified scene
document.getElementById('btn-stop').onclick = () => setMode('edit');

on('simStatus', () => {
  playBtn.disabled = sim.state !== 'stopped';
  pauseBtn.disabled = sim.state === 'stopped';
  pauseBtn.textContent = sim.state === 'paused' ? 'Resume' : 'Pause';
  stepBtn.disabled = sim.state !== 'paused';
});
