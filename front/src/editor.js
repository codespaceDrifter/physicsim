// Edit mode: picking, transform gizmo, add / insert / joint / duplicate / delete actions, keyboard shortcuts.
import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { app, on, emit, select, clickSelect } from './state.js';
import { renderer, scene3d, camera, controls, viewHelper, frameObjects } from './viewport.js';
import { objectsById, pickables, bodyGroups } from './sceneView.js';
import { findItem, uniqueId, makeBody, LIST_KEY, DEFAULT_SHAPES } from './model.js';
import { getJson } from './api.js';
import { toast } from './toast.js';

const canvas = renderer.domElement;

const gizmo = new TransformControls(camera, canvas);
gizmo.setSize(0.8);
scene3d.add(gizmo.getHelper());
on('camera', () => { gizmo.camera = camera; });

gizmo.addEventListener('dragging-changed', (e) => {
  controls.enabled = !e.value;
  // drag finished: rebuild views so joint markers etc. match the moved body
  if (!e.value) emit('scene', 'gizmo');
});

// write the dragged pose back into the scene JSON
gizmo.addEventListener('objectChange', () => {
  const body = findItem(app.scene, 'body', gizmo.object.userData.bodyId);
  const p = gizmo.object.position;
  const q = gizmo.object.quaternion;
  body.position = [p.x, p.y, p.z];
  body.rotation = [q.w, q.x, q.y, q.z];
  emit('pose');
});

// attach the gizmo to the primary selected body (edit mode only)
export function attachGizmo() {
  const primary = app.selection.at(-1);
  if (app.mode !== 'edit' || primary?.kind !== 'body') {
    gizmo.detach();
    return;
  }
  const group = objectsById.body.get(primary.id);
  group.userData.bodyId = primary.id;
  gizmo.attach(group);
}

export function setGizmoMode(mode) {
  gizmo.setMode(mode);
  document.getElementById('btn-translate').classList.toggle('on', mode === 'translate');
  document.getElementById('btn-rotate').classList.toggle('on', mode === 'rotate');
}

export function toggleSnap() {
  const snapping = gizmo.translationSnap === null;
  gizmo.setTranslationSnap(snapping ? 0.01 : null);
  gizmo.setRotationSnap(snapping ? THREE.MathUtils.degToRad(15) : null);
  document.getElementById('btn-snap').classList.toggle('on', snapping);
}

// ---------- picking: a click (not a drag) selects whatever is under the cursor ----------
const raycaster = new THREE.Raycaster();
raycaster.params.Line.threshold = 0.01;
let downX = 0;
let downY = 0;
let downOnGizmo = false; // pointer went down on a gizmo handle -> not a selection click

canvas.addEventListener('pointerdown', (e) => {
  downX = e.clientX;
  downY = e.clientY;
  downOnGizmo = gizmo.object !== undefined && gizmo.axis !== null;
});

canvas.addEventListener('pointerup', (e) => {
  if (e.button !== 0 || downOnGizmo || Math.hypot(e.clientX - downX, e.clientY - downY) > 4) return;
  if (viewHelper.handleClick(e)) return;
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObjects(pickables, false);
  // joint / camera markers are drawn on top of bodies, so they win over the body they sit in
  const hit = hits.find(h => h.object.userData.kind !== 'body') ?? hits[0];
  clickSelect(hit ? { kind: hit.object.userData.kind, id: hit.object.userData.id } : null, e.shiftKey);
});

// frame the selection, or every non-plane body when nothing is selected
export function frameSelection() {
  const objects = app.selection.length
    ? app.selection.map(s => objectsById[s.kind].get(s.id))
    : bodyGroups;
  frameObjects(objects.filter(o => !o.userData.isPlane));
}

// ---------- actions ----------
export function addPrimitive(type) {
  // drop it under the orbit target, resting on y = 0
  const x = Math.round(controls.target.x * 100) / 100;
  const z = Math.round(controls.target.z * 100) / 100;
  const body = makeBody(uniqueId(app.scene.bodies, type), structuredClone(DEFAULT_SHAPES[type]), [x, 0.1, z]);
  app.scene.bodies.push(body);
  app.selection = [{ kind: 'body', id: body.id }];
  emit('scene', 'editor');
}

export async function insertLekiwi() {
  const prefab = await getJson('/prefabs/lekiwi');
  // first free namespace: lekiwi/, lekiwi2/, lekiwi3/, ...
  let prefix = 'lekiwi/';
  for (let n = 2; app.scene.bodies.some(b => b.id.startsWith(prefix)); n++) prefix = `lekiwi${n}/`;
  const rename = (id) => {
    if (!id.startsWith('lekiwi/')) throw new Error(`prefab id "${id}" is not under lekiwi/`);
    return prefix + id.slice('lekiwi/'.length);
  };
  // place it 0.6 m beyond the farthest existing body along +z so it doesn't overlap anything
  const others = app.scene.bodies.filter(b => b.shape.type !== 'plane');
  const offsetZ = others.length ? Math.max(...others.map(b => b.position[2])) + 0.6 : 0;

  for (const body of prefab.bodies) {
    body.id = rename(body.id);
    body.position[2] += offsetZ;
    app.scene.bodies.push(body);
  }
  for (const joint of prefab.joints) {
    joint.id = rename(joint.id);
    joint.body_a = rename(joint.body_a);
    joint.body_b = rename(joint.body_b);
    joint.anchor[2] += offsetZ;
    app.scene.joints.push(joint);
  }
  for (const cam of prefab.cameras) {
    cam.id = rename(cam.id);
    cam.body = rename(cam.body);
    app.scene.cameras.push(cam);
  }
  app.selection = [{ kind: 'body', id: `${prefix}chassis` }];
  emit('scene', 'editor');
  frameSelection();
}

export function createJoint(type) {
  const bodies = app.selection.filter(s => s.kind === 'body'); // [{kind, id}]
  if (bodies.length !== 2) {
    toast('Select exactly two bodies (shift-click) to create a joint');
    return;
  }
  const [a, b] = bodies;
  const joint = {
    id: uniqueId(app.scene.joints, 'joint'),
    type,
    body_a: a.id,
    body_b: b.id,
    anchor: [...findItem(app.scene, 'body', b.id).position],
    axis: [0, 0, 1],
    limits: null,
    drive: null,
  };
  app.scene.joints.push(joint);
  app.selection = [{ kind: 'joint', id: joint.id }];
  emit('scene', 'editor');
}

export function duplicateSelection() {
  const copies = app.selection.map(({ kind, id }) => {
    const list = app.scene[LIST_KEY[kind]];
    const copy = structuredClone(findItem(app.scene, kind, id));
    copy.id = uniqueId(list, id);
    if (kind === 'body') copy.position[0] += 0.25;
    list.push(copy);
    return { kind, id: copy.id };
  }); // [{kind, id}]
  app.selection = copies;
  emit('scene', 'editor');
}

// deleting a body also deletes the joints and cameras that reference it
export function deleteSelection() {
  const gone = new Set(app.selection.map(s => `${s.kind}:${s.id}`)); // {"kind:id"}
  const scene = app.scene;
  scene.bodies = scene.bodies.filter(b => !gone.has(`body:${b.id}`));
  const bodyIds = new Set(scene.bodies.map(b => b.id)); // {bodyId} remaining bodies
  scene.joints = scene.joints.filter(j => !gone.has(`joint:${j.id}`) && bodyIds.has(j.body_a) && bodyIds.has(j.body_b));
  scene.cameras = scene.cameras.filter(c => !gone.has(`camera:${c.id}`) && bodyIds.has(c.body));
  app.selection = [];
  emit('scene', 'editor');
}

// ---------- keyboard shortcuts ----------
window.addEventListener('keydown', (e) => {
  if (e.target.closest('input, select, textarea')) return;
  const key = e.key.toLowerCase();
  if (key === 'f') frameSelection();
  if (app.mode !== 'edit') return;
  if (key === 'delete' || key === 'backspace') deleteSelection();
  else if (key === 'd' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    duplicateSelection();
  } else if (key === 't') setGizmoMode('translate');
  else if (key === 'r' && !e.ctrlKey && !e.metaKey) setGizmoMode('rotate');
  else if (key === 'escape') select([]);
});
