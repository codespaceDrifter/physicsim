// Renderer, cameras (perspective / orthographic), orbit controls, lights, view cube, render loop.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ViewHelper } from 'three/addons/helpers/ViewHelper.js';
import { emit } from './state.js';

const container = document.getElementById('viewport');
const BACKGROUND = 0x25282d;

export const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.autoClear = false; // we draw the scene and then the view cube on top
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.NeutralToneMapping;
container.prepend(renderer.domElement);

export const scene3d = new THREE.Scene();
scene3d.background = new THREE.Color(BACKGROUND);
// hides the edge of the large ground plane at the horizon
scene3d.fog = new THREE.Fog(BACKGROUND, 40, 150);

scene3d.add(new THREE.HemisphereLight(0xdfe7f2, 0x30333a, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(3, 6, 2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -6;
sun.shadow.camera.right = 6;
sun.shadow.camera.top = 6;
sun.shadow.camera.bottom = -6;
sun.shadow.camera.far = 30;
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.01;
sun.shadow.radius = 3;
scene3d.add(sun);

const perspCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 500);
perspCamera.position.set(1.6, 1.2, 1.6);
const orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 500);

// active camera, swapped by setOrthographic()
export let camera = perspCamera;

export const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.target.set(0, 0.1, 0);
controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
// OrbitControls only rotates. its zoom and pan scale with the distance to the orbit target (slow near it), so zoom and
// pan are done below at a constant speed instead
controls.enableZoom = false;
controls.enablePan = false;

// ---------- constant-speed zoom (wheel) and pan (ctrl + left drag, or right drag) ----------
let navScale = 1; // m, size of the shown scene: sets zoom / pan speed, constant within a scene
const ZOOM_STEP = 0.06; // fraction of navScale moved per wheel notch (deltaY 100)
const PAN_PER_PIXEL = 0.0012; // fraction of navScale moved per pixel dragged
const MIN_PIVOT_DISTANCE = 0.02; // fraction of navScale: zooming closer than this pushes the pivot forward

export function setNavigationScale(size) {
  navScale = size;
}

renderer.domElement.addEventListener('wheel', (e) => {
  if (!controls.enabled) return;
  e.preventDefault();
  // > 0 = zoom in. wheel notches are deltaY ±100, trackpads send small deltas: both move proportionally
  const step = -e.deltaY / 100 * ZOOM_STEP * navScale;
  if (camera === orthoCamera) {
    // orthographic: shrink / grow the visible half height by the same constant amount
    const half = orthoCamera.top / orthoCamera.zoom;
    orthoCamera.zoom = orthoCamera.top / Math.max(half - step, MIN_PIVOT_DISTANCE * navScale);
    orthoCamera.updateProjectionMatrix();
    return;
  }
  const dir = controls.target.clone().sub(camera.position).normalize();
  camera.position.addScaledVector(dir, step);
  // passed (or nearly reached) the pivot: carry it forward so zooming in never stops
  const minDist = MIN_PIVOT_DISTANCE * navScale;
  if (controls.target.clone().sub(camera.position).dot(dir) < minDist) {
    controls.target.copy(camera.position).addScaledVector(dir, minDist);
  }
}, { passive: false });

let panning = null; // {x, y} last pointer position while panning, null otherwise
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (!controls.enabled || !(e.button === 2 || (e.button === 0 && (e.ctrlKey || e.metaKey)))) return;
  panning = { x: e.clientX, y: e.clientY };
});
window.addEventListener('pointermove', (e) => {
  if (panning === null) return;
  const dx = e.clientX - panning.x;
  const dy = e.clientY - panning.y;
  panning = { x: e.clientX, y: e.clientY };
  // move camera and pivot together in the screen plane: the scene follows the cursor at constant speed
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  const offset = right.multiplyScalar(-dx).addScaledVector(up, dy).multiplyScalar(PAN_PER_PIXEL * navScale);
  camera.position.add(offset);
  controls.target.add(offset);
});
window.addEventListener('pointerup', () => {
  panning = null;
});

// clickable axes indicator in the bottom-right corner, rebuilt when the camera changes
export let viewHelper = null;
function makeViewHelper() {
  viewHelper?.dispose();
  viewHelper = new ViewHelper(camera, renderer.domElement);
  viewHelper.center = controls.target;
  viewHelper.setLabels('X', 'Y', 'Z');
  viewHelper.setLabelStyle('bold 22px system-ui', '#ffffff', 16);
}
makeViewHelper();

// size the orthographic frustum so it shows what the perspective camera shows at the orbit target
function fitOrtho() {
  const distance = camera.position.distanceTo(controls.target);
  const halfHeight = distance * Math.tan(THREE.MathUtils.degToRad(perspCamera.fov / 2));
  const aspect = container.clientWidth / container.clientHeight;
  orthoCamera.top = halfHeight;
  orthoCamera.bottom = -halfHeight;
  orthoCamera.left = -halfHeight * aspect;
  orthoCamera.right = halfHeight * aspect;
  orthoCamera.zoom = 1;
  orthoCamera.updateProjectionMatrix();
}

export function setOrthographic(ortho) {
  const next = ortho ? orthoCamera : perspCamera;
  next.position.copy(camera.position);
  next.quaternion.copy(camera.quaternion);
  camera = next;
  controls.object = camera;
  if (ortho) fitOrtho();
  makeViewHelper();
  controls.update();
  emit('camera');
}

// standard CAD views: camera direction from the orbit target
const VIEW_DIRS = {
  top: [0, 1, 0.0001],
  front: [0, 0, 1],
  right: [1, 0, 0],
  iso: [1, 1, 1],
};

export function setStandardView(name) {
  const dir = VIEW_DIRS[name];
  if (!dir) throw new Error(`unknown view "${name}"`);
  const distance = camera.position.distanceTo(controls.target);
  camera.position.set(...dir).normalize().multiplyScalar(distance).add(controls.target);
  controls.update();
}

// zoom so all given objects fit, keeping the viewing direction
export function frameObjects(objects) {
  if (objects.length === 0) return;
  const box = new THREE.Box3();
  for (const obj of objects) box.expandByObject(obj);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius, 0.05);
  const distance = (radius / Math.sin(THREE.MathUtils.degToRad(perspCamera.fov / 2))) * 1.15;
  const dir = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(dir, distance);
  if (camera === orthoCamera) fitOrtho();
  controls.update();
}

// keep renderer + both cameras matched to the viewport size
new ResizeObserver(() => {
  const w = container.clientWidth;
  const h = container.clientHeight;
  renderer.setSize(w, h);
  perspCamera.aspect = w / h;
  perspCamera.updateProjectionMatrix();
  orthoCamera.left = orthoCamera.bottom * (w / h);
  orthoCamera.right = orthoCamera.top * (w / h);
  orthoCamera.updateProjectionMatrix();
}).observe(container);

// per-frame callbacks (playback clock, teleop), called with dt in seconds
const tickFns = []; // [(dt) => void]
export function onTick(fn) {
  tickFns.push(fn);
}

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = clock.getDelta();
  if (viewHelper.animating) viewHelper.update(dt);
  else controls.update();
  for (const fn of tickFns) fn(dt);
  renderer.clear();
  renderer.render(scene3d, camera);
  viewHelper.render(renderer);
});
