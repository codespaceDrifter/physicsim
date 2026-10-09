// Simulate-mode right panel: sim state/time, joint readout, ToFuBot keyboard teleop + autopilot, camera feeds.
import { app, on } from './state.js';
import { findItem } from './model.js';
import { onTick } from './viewport.js';
import { sim, send } from './simClient.js';

const stateEl = document.getElementById('sim-state');
const timeEl = document.getElementById('sim-time');
const statusTimeEl = document.getElementById('status-time');
const robotSection = document.getElementById('robot-section');
const autopilotSelect = document.getElementById('autopilot');
const gripperEl = document.getElementById('gripper-state');
const jointReadout = document.getElementById('joint-readout');
const feedsEl = document.getElementById('feeds');

const P = 'tofubot/'; // only the first ToFuBot (namespace tofubot/) is teleoperated
const DRIVE_SPEED = 15; // rad/s rear wheel velocity target while W/S held
const STEER_ANGLE = 0.5; // rad front wheel steer while A/D held, positive = left turn
const ARM_RATE = 1.2; // rad/s arm joint target change while key held
const GRIPPER_OPEN = 0.035; // m finger target when open (closed = 0)
const SEND_INTERVAL = 0.05; // s between drive messages (20 Hz)
// key -> [joint name, direction] for incremental arm controls
const ARM_KEYS = {
  j: ['turret', 1], l: ['turret', -1],
  i: ['shoulder', 1], k: ['shoulder', -1],
  u: ['elbow', 1], o: ['elbow', -1],
}; // {key: [joint name, +1 | -1]}

const held = new Set(); // {key} teleop keys currently held, lowercase
let hasRobot = false; // shown scene contains tofubot/ joints
let targets = {}; // {jointId: number} current teleop drive targets
let lastSent = {}; // {jointId: number} targets the server already has
let gripperOpen = false;
let sinceSend = 0; // s since the last drive message
const jointRows = new Map(); // {jointId: <span> value cell}
const feeds = new Map(); // {cameraId: {canvas, ctx, image: ImageData}}

// reset teleop targets to the scene's initial drive targets, clear readouts and feeds
on('simPlay', () => {
  targets = {};
  if (hasRobot) {
    for (const name of ['steer_fl', 'steer_fr', 'drive_rl', 'drive_rr', 'turret', 'shoulder', 'elbow', 'finger_l', 'finger_r']) {
      const joint = findItem(app.scene, 'joint', P + name);
      if (joint.drive === null) throw new Error(`ToFuBot joint "${joint.id}" has no drive`);
      targets[joint.id] = joint.drive.target;
    }
    gripperOpen = targets[`${P}finger_l`] > GRIPPER_OPEN / 2;
  }
  lastSent = { ...targets };
  jointReadout.innerHTML = '';
  jointRows.clear();
  feedsEl.innerHTML = '';
  feeds.clear();
  gripperEl.textContent = gripperOpen ? 'open' : 'closed';
});

on('mode', () => {
  hasRobot = app.mode === 'simulate' && app.scene.joints.some(j => j.id.startsWith(P));
  robotSection.style.display = hasRobot ? '' : 'none';
  held.clear();
  if (app.mode !== 'simulate') {
    feedsEl.innerHTML = '';
    feeds.clear();
  }
});

on('simStatus', () => {
  stateEl.textContent = sim.state;
  autopilotSelect.value = sim.autopilot;
  // while the autopilot drives, our last sent targets go stale -> resend everything once it's off
  if (sim.autopilot !== 'off') lastSent = {};
});

autopilotSelect.onchange = () => send({ type: 'autopilot', mode: autopilotSelect.value });

on('simFrame', (msg) => {
  timeEl.textContent = `${msg.time.toFixed(3)} s`;
  statusTimeEl.textContent = `t = ${msg.time.toFixed(3)} s`;

  for (const [id, value] of Object.entries(msg.joints)) {
    if (!jointRows.has(id)) {
      const row = document.createElement('div');
      row.className = 'joint-row';
      row.innerHTML = '<span></span><span></span>';
      row.firstChild.textContent = id;
      row.firstChild.title = id;
      jointReadout.append(row);
      jointRows.set(id, row.lastChild);
    }
    const unit = findItem(app.scene, 'joint', id).type === 'prismatic' ? 'm' : 'rad';
    jointRows.get(id).textContent = `${value.toFixed(3)} ${unit}`;
  }

  for (const cam of msg.cameras) {
    if (!feeds.has(cam.id)) {
      const panel = document.createElement('div');
      panel.className = 'feed';
      panel.innerHTML = '<div class="feed-title"></div><canvas></canvas>';
      panel.firstChild.textContent = cam.id;
      const canvas = panel.lastChild;
      canvas.width = cam.width;
      canvas.height = cam.height;
      canvas.style.width = `${cam.width * 4}px`;
      canvas.style.height = `${cam.height * 4}px`;
      feedsEl.append(panel);
      const ctx = canvas.getContext('2d');
      feeds.set(cam.id, { canvas, ctx, image: ctx.createImageData(cam.width, cam.height) });
    }
    const { ctx, image } = feeds.get(cam.id);
    // base64 RGB (row 0 = top, same as canvas) -> RGBA ImageData
    const rgb = atob(cam.rgb);
    if (rgb.length !== cam.width * cam.height * 3 || image.width !== cam.width || image.height !== cam.height) {
      throw new Error(`camera "${cam.id}": got ${rgb.length} bytes for ${cam.width}x${cam.height} image`);
    }
    const px = image.data;
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
      px[j] = rgb.charCodeAt(i);
      px[j + 1] = rgb.charCodeAt(i + 1);
      px[j + 2] = rgb.charCodeAt(i + 2);
      px[j + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }
});

// ---------- keyboard teleop ----------
const TELEOP_KEYS = new Set(['w', 'a', 's', 'd', ...Object.keys(ARM_KEYS)]); // {key}

window.addEventListener('keydown', (e) => {
  if (!hasRobot || e.target.closest('input, select, textarea') || e.ctrlKey || e.metaKey) return;
  const key = e.key.toLowerCase();
  if (e.code === 'Space') {
    e.preventDefault();
    if (e.repeat) return;
    gripperOpen = !gripperOpen;
    gripperEl.textContent = gripperOpen ? 'open' : 'closed';
  } else if (TELEOP_KEYS.has(key)) {
    held.add(key);
  }
});
window.addEventListener('keyup', (e) => held.delete(e.key.toLowerCase()));
window.addEventListener('blur', () => held.clear());

onTick((dt) => {
  if (!hasRobot || sim.state !== 'running' || sim.autopilot !== 'off') return;

  // held -> fixed value, released -> 0
  const drive = (held.has('w') ? DRIVE_SPEED : 0) - (held.has('s') ? DRIVE_SPEED : 0);
  const steer = (held.has('a') ? STEER_ANGLE : 0) - (held.has('d') ? STEER_ANGLE : 0);
  targets[`${P}drive_rl`] = drive;
  targets[`${P}drive_rr`] = drive;
  targets[`${P}steer_fl`] = steer;
  targets[`${P}steer_fr`] = steer;
  // held -> target moves at ARM_RATE, clamped to the joint's limits
  for (const [key, [name, dir]] of Object.entries(ARM_KEYS)) {
    if (!held.has(key)) continue;
    const id = P + name;
    const limits = findItem(app.scene, 'joint', id).limits;
    let value = targets[id] + dir * ARM_RATE * dt;
    if (limits !== null) value = Math.min(Math.max(value, limits[0]), limits[1]);
    targets[id] = value;
  }
  targets[`${P}finger_l`] = gripperOpen ? GRIPPER_OPEN : 0.0;
  targets[`${P}finger_r`] = gripperOpen ? GRIPPER_OPEN : 0.0;

  sinceSend += dt;
  if (sinceSend < SEND_INTERVAL) return;
  sinceSend = 0;
  const changes = Object.fromEntries(Object.entries(targets).filter(([id, v]) => lastSent[id] !== v)); // {jointId: target}
  if (Object.keys(changes).length === 0) return;
  send({ type: 'drive', targets: changes });
  Object.assign(lastSent, changes);
});
