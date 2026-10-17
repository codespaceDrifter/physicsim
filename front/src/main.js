// Entry point: wires the modules together through the event bus and loads the first scene.
import * as THREE from 'three';
import { app, on, emit } from './state.js';
import { setNavigationScale } from './viewport.js';
import { getJson } from './api.js';
import { newScene } from './model.js';
import { toast } from './toast.js';
import { rebuild, highlight } from './sceneView.js';
import { attachGizmo } from './editor.js';
import { render as renderOutliner } from './outliner.js';
import { render as renderInspector } from './inspector.js';
import { refreshSceneList, loadScene, setScene } from './toolbar.js';
import './simClient.js';
import './robotPanel.js';
import './playback.js';
import { startGames } from './demo.js';

const statusMode = document.getElementById('status-mode');
const statusSel = document.getElementById('status-sel');
const statusBackend = document.getElementById('status-backend');

// every uncaught error ends up visible as a toast (and in the console with its stack)
window.addEventListener('error', (e) => toast(e.message));
window.addEventListener('unhandledrejection', (e) => toast(e.reason?.message ?? String(e.reason)));

function updateStatus() {
  statusMode.textContent = `${app.mode} · ${app.shown?.name ?? 'no scene'}`;
  const sel = app.selection;
  statusSel.textContent = sel.length === 0 ? '' : sel.length === 1 ? `${sel[0].kind} ${sel[0].id}` : `${sel.length} selected`;
}

on('scene', (source) => {
  rebuild(app.shown);
  // zoom / pan speed follows the scene's size: spread of the non-plane bodies (a 1 m robot scene vs 700 m of stars)
  const points = app.shown.bodies.filter(b => b.shape.type !== 'plane').map(b => new THREE.Vector3(...b.position)); // [Vector3]
  const size = points.length > 0 ? new THREE.Box3().setFromPoints(points).getSize(new THREE.Vector3()).length() : 1;
  setNavigationScale(Math.max(size, 0.5));
  attachGizmo();
  renderOutliner();
  // the inspector already shows what it just wrote; re-rendering would steal input focus
  if (source !== 'inspector') renderInspector();
  updateStatus();
});

on('selection', () => {
  highlight();
  attachGizmo();
  renderOutliner();
  renderInspector();
  updateStatus();
});

on('pose', () => renderInspector());

on('mode', (prev) => {
  document.body.dataset.mode = app.mode;
  // edit <-> simulate share the edited scene (playback.js handles entering playback)
  if (app.mode !== 'playback') {
    app.shown = app.scene;
    if (prev === 'playback') app.selection = [];
    emit('scene', 'mode');
  }
  if (app.mode === 'simulate') emit('simStatus');
  document.getElementById('status-time').textContent = '';
  updateStatus();
});

getJson('/health')
  .then(() => {
    statusBackend.textContent = 'backend: online';
    statusBackend.className = 'online';
  })
  .catch(() => {
    statusBackend.textContent = 'backend: offline';
    statusBackend.className = 'offline';
  });

// start with an empty scene so the editor works even without the backend, then open the first saved scene for the
// editor and start the simple view's pre-recorded games
setScene(newScene('untitled'));
refreshSceneList().then(async (scenes) => {
  if (scenes.length > 0) await loadScene(scenes[0]);
  await startGames();
});
