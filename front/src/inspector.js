// Right panel: edits every schema field of the primary selection (or scene settings when nothing is selected).
// Inputs write straight into the scene JSON, then emit('scene', 'inspector') so the views rebuild.
import * as THREE from 'three';
import { app, emit } from './state.js';
import { findItem, LIST_KEY, DEFAULT_SHAPES } from './model.js';
import { toast } from './toast.js';

const root = document.getElementById('inspector');
let form = null; // <fieldset> the current render writes rows into (disabled outside edit mode)
let refreshDerived = null; // () => void, updates the volume / mass readout of the shown body, null otherwise

// short display of a number: 6 significant digits, no trailing zeros
function fmt(v) {
  return String(Number(v.toPrecision(6)));
}

// a value was written into the scene JSON
function changed() {
  emit('scene', 'inspector');
  refreshDerived?.();
}

function section(title) {
  const el = document.createElement('div');
  el.className = 'section';
  el.innerHTML = '<div class="section-title"></div>';
  el.firstChild.textContent = title;
  form.append(el);
  return el;
}

function row(parent, label, ...controls) {
  const el = document.createElement('div');
  el.className = 'row';
  const labelEl = document.createElement('label');
  labelEl.textContent = label;
  const fields = document.createElement('div');
  fields.className = 'fields';
  fields.append(...controls);
  el.append(labelEl, fields);
  parent.append(el);
  return fields;
}

function numberInput(value, onCommit, integer = false) {
  const input = document.createElement('input');
  input.type = 'number';
  input.step = 'any';
  input.value = fmt(value);
  input.onchange = () => {
    const v = Number(input.value);
    if (input.value.trim() === '' || !Number.isFinite(v) || (integer && !Number.isInteger(v))) {
      toast(`"${input.value}" is not a valid ${integer ? 'integer' : 'number'}`);
      input.value = fmt(value);
      return;
    }
    value = v;
    onCommit(v);
    changed();
  };
  return input;
}

function num(parent, label, value, onCommit, integer = false) {
  row(parent, label, numberInput(value, onCommit, integer));
}

// x/y/z inputs that write into arr in place; after() runs after each write
function vec(parent, label, arr, after = null) {
  const inputs = arr.map((v, i) => numberInput(v, (x) => {
    arr[i] = x;
    after?.();
  }));
  row(parent, label, ...inputs).classList.add('vec');
}

// quaternion [w, x, y, z] shown as XYZ euler degrees
function euler(parent, label, quat, onCommit) {
  const e = new THREE.Euler().setFromQuaternion(new THREE.Quaternion(quat[1], quat[2], quat[3], quat[0]), 'XYZ');
  const degrees = [e.x, e.y, e.z].map(THREE.MathUtils.radToDeg); // [x, y, z] degrees
  const inputs = degrees.map((d, i) => numberInput(d, (x) => {
    degrees[i] = x;
    const [rx, ry, rz] = degrees.map(THREE.MathUtils.degToRad);
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz, 'XYZ'));
    onCommit([q.w, q.x, q.y, q.z]);
  }));
  row(parent, label, ...inputs).classList.add('vec');
}

function check(parent, label, value, onCommit) {
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = value;
  input.onchange = () => {
    onCommit(input.checked);
    changed();
  };
  row(parent, label, input);
}

function choice(parent, label, value, options, onCommit) {
  const input = document.createElement('select');
  for (const opt of options) input.add(new Option(opt, opt));
  input.value = value;
  input.onchange = () => {
    onCommit(input.value);
    changed();
  };
  row(parent, label, input);
}

function readout(parent, label) {
  const span = document.createElement('span');
  span.className = 'readout';
  row(parent, label, span);
  return span;
}

export function render() {
  root.innerHTML = '<div class="panel-title">Properties<span class="sub"></span></div>';
  refreshDerived = null;
  form = document.createElement('fieldset');
  form.disabled = app.mode !== 'edit';
  root.append(form);
  const scene = app.shown;
  if (!scene) {
    form.innerHTML = '<div class="note">No scene loaded.</div>';
    return;
  }
  const primary = app.selection.at(-1);

  // ---------- scene settings ----------
  if (!primary) {
    root.querySelector('.sub').textContent = 'Scene';
    const sec = section('Scene settings');
    const nameInput = document.createElement('input');
    nameInput.value = scene.name;
    nameInput.onchange = () => {
      scene.name = nameInput.value.trim();
      changed();
    };
    row(sec, 'Name', nameInput);
    num(sec, 'dt (s)', scene.settings.dt, (v) => { scene.settings.dt = v; });
    num(sec, 'Substeps', scene.settings.substeps, (v) => { scene.settings.substeps = v; }, true);
    vec(sec, 'Gravity (m/s²)', scene.settings.gravity);
    num(sec, 'G', scene.settings.G, (v) => { scene.settings.G = v; });
    return;
  }

  const item = findItem(scene, primary.kind, primary.id);
  root.querySelector('.sub').textContent = item.id;
  const head = section(primary.kind[0].toUpperCase() + primary.kind.slice(1));
  if (app.selection.length > 1) {
    head.insertAdjacentHTML('beforeend', `<div class="note" style="padding:0 0 6px">${app.selection.length} selected, showing the last one</div>`);
  }

  // id: renaming a body rewrites the joint / camera references to it
  const idInput = document.createElement('input');
  idInput.value = item.id;
  idInput.onchange = () => {
    const newId = idInput.value.trim();
    if (newId === item.id) return;
    if (!newId || scene[LIST_KEY[primary.kind]].some(x => x.id === newId)) {
      toast(newId ? `${primary.kind} id "${newId}" already exists` : 'id cannot be empty');
      idInput.value = item.id;
      return;
    }
    if (primary.kind === 'body') {
      for (const joint of scene.joints) {
        if (joint.body_a === item.id) joint.body_a = newId;
        if (joint.body_b === item.id) joint.body_b = newId;
      }
      for (const cam of scene.cameras) {
        if (cam.body === item.id) cam.body = newId;
      }
    }
    item.id = newId;
    primary.id = newId;
    emit('scene', 'rename');
  };
  row(head, 'Id', idInput);

  // ---------- body ----------
  if (primary.kind === 'body') {
    const body = item;
    const shape = body.shape;
    choice(head, 'Shape', shape.type, Object.keys(DEFAULT_SHAPES), (type) => {
      body.shape = structuredClone(DEFAULT_SHAPES[type]);
      if (type === 'plane') body.static = true;
      render();
    });
    if (shape.type === 'box') vec(head, 'Size (m)', shape.size);
    if (shape.type === 'sphere' || shape.type === 'cylinder') num(head, 'Radius (m)', shape.radius, (v) => { shape.radius = v; });
    if (shape.type === 'cylinder') num(head, 'Height (m)', shape.height, (v) => { shape.height = v; });

    const pose = section('Initial state');
    vec(pose, 'Position (m)', body.position);
    euler(pose, 'Rotation (°)', body.rotation, (q) => { body.rotation = q; });
    vec(pose, 'Velocity (m/s)', body.velocity);
    vec(pose, 'Ang. vel (rad/s)', body.angular_velocity);

    const phys = section('Physics');
    num(phys, 'Density (kg/m³)', body.density, (v) => { body.density = v; });
    check(phys, 'Static', body.static, (v) => { body.static = v; });
    check(phys, 'Collide', body.collide, (v) => { body.collide = v; });
    num(phys, 'Group', body.group, (v) => { body.group = v; }, true);
    num(phys, 'Friction', body.friction, (v) => { body.friction = v; });
    num(phys, 'Restitution', body.restitution, (v) => { body.restitution = v; });
    const colorInput = document.createElement('input');
    colorInput.type = 'color';
    colorInput.value = body.color;
    colorInput.oninput = () => {
      body.color = colorInput.value;
      changed();
    };
    row(phys, 'Color', colorInput);
    const volumeEl = readout(phys, 'Volume');
    const massEl = readout(phys, 'Mass');
    refreshDerived = () => {
      const s = body.shape;
      let volume = Infinity; // plane: infinite
      if (s.type === 'box') volume = s.size[0] * s.size[1] * s.size[2];
      if (s.type === 'sphere') volume = (4 / 3) * Math.PI * s.radius ** 3;
      if (s.type === 'cylinder') volume = Math.PI * s.radius ** 2 * s.height;
      volumeEl.textContent = volume === Infinity ? '∞' : `${fmt(volume)} m³`;
      massEl.textContent = body.static ? '∞ (static)' : `${fmt(volume * body.density)} kg`;
    };
    refreshDerived();
  }

  // ---------- joint ----------
  if (primary.kind === 'joint') {
    const joint = item;
    const bodyIds = scene.bodies.map(b => b.id); // [bodyId]
    // revolute values are angles, prismatic values are lengths
    const unit = joint.type === 'prismatic' ? 'm' : 'rad';
    choice(head, 'Type', joint.type, ['revolute', 'prismatic', 'fixed'], (v) => {
      joint.type = v;
      render();
    });
    choice(head, 'Body A', joint.body_a, bodyIds, (v) => { joint.body_a = v; });
    choice(head, 'Body B', joint.body_b, bodyIds, (v) => { joint.body_b = v; });

    const geo = section('Geometry (world, at definition)');
    vec(geo, 'Anchor (m)', joint.anchor);
    vec(geo, 'Axis', joint.axis, () => {
      const len = Math.hypot(...joint.axis);
      if (len === 0) {
        toast('joint axis cannot be zero, reset to [0, 0, 1]');
        joint.axis = [0, 0, 1];
      } else {
        joint.axis = joint.axis.map(a => a / len);
      }
      render();
    });

    const lim = section('Limits');
    check(lim, 'Enabled', joint.limits !== null, (on) => {
      joint.limits = on ? [-1, 1] : null;
      render();
    });
    if (joint.limits !== null) {
      num(lim, `Lower (${unit})`, joint.limits[0], (v) => { joint.limits[0] = v; });
      num(lim, `Upper (${unit})`, joint.limits[1], (v) => { joint.limits[1] = v; });
    }

    const drv = section('Drive');
    check(drv, 'Enabled', joint.drive !== null, (on) => {
      joint.drive = on ? { mode: 'position', stiffness: 20.0, damping: 1.0, max_force: 5.0, target: 0.0 } : null;
      render();
    });
    if (joint.drive !== null) {
      const drive = joint.drive;
      choice(drv, 'Mode', drive.mode, ['position', 'velocity'], (v) => {
        drive.mode = v;
        render();
      });
      num(drv, 'Stiffness', drive.stiffness, (v) => { drive.stiffness = v; });
      num(drv, 'Damping', drive.damping, (v) => { drive.damping = v; });
      num(drv, 'Max force', drive.max_force, (v) => { drive.max_force = v; });
      num(drv, `Target (${unit}${drive.mode === 'velocity' ? '/s' : ''})`, drive.target, (v) => { drive.target = v; });
    }
  }

  // ---------- camera ----------
  if (primary.kind === 'camera') {
    const cam = item;
    choice(head, 'Body', cam.body, scene.bodies.map(b => b.id), (v) => { cam.body = v; });
    const mount = section('Mount (body local frame)');
    vec(mount, 'Position (m)', cam.position);
    euler(mount, 'Rotation (°)', cam.rotation, (q) => { cam.rotation = q; });
    const img = section('Image');
    num(img, 'FOV (° vert.)', cam.fov, (v) => { cam.fov = v; });
    num(img, 'Width (px)', cam.width, (v) => { cam.width = v; }, true);
    num(img, 'Height (px)', cam.height, (v) => { cam.height = v; }, true);
  }
}
