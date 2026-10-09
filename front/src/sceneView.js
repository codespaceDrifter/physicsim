// Scene JSON -> Three.js objects. Everything here is a view of the JSON and is rebuilt from it.
import * as THREE from 'three';
import { scene3d } from './viewport.js';
import { app } from './state.js';
import { findItem } from './model.js';

const root = new THREE.Group();
scene3d.add(root);

export const bodyGroups = []; // [THREE.Group] in scene["bodies"] order, pose = body pose
export const objectsById = { body: new Map(), joint: new Map(), camera: new Map() }; // {kind: {id: THREE.Object3D}}
export const pickables = []; // [THREE.Object3D] raycast targets, userData = {kind, id}

const SELECT_COLOR = 0x3d9bff;
const EDGE_COLOR = 0x15171a;
const JOINT_COLORS = { revolute: 0xff9f1c, prismatic: 0x2ec4b6, fixed: 0xa0a6ae }; // {joint type: color}
const CAMERA_COLOR = 0xc77dff;

// CAD grid drawn on plane bodies: minor lines every 0.1 m, major every 1 m, red x / blue z axis, fades out with distance
const GRID_VERTEX = /* glsl */ `
  varying vec3 vWorld;
  varying vec2 vLocal;
  void main() {
    vLocal = position.xy;
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }`;
const GRID_FRAGMENT = /* glsl */ `
  varying vec3 vWorld;
  varying vec2 vLocal;
  // anti-aliased line strength (0..1) of a grid with the given spacing, faded out once cells get too small on screen
  float grid(float spacing) {
    vec2 cell = vLocal / spacing;
    vec2 perPixel = fwidth(cell);
    vec2 g = abs(fract(cell - 0.5) - 0.5) / perPixel;
    float line = 1.0 - min(min(g.x, g.y), 1.0);
    return line * (1.0 - smoothstep(0.08, 0.25, max(perPixel.x, perPixel.y)));
  }
  float axisLine(float coord) {
    return 1.0 - min(abs(coord) / (fwidth(coord) * 1.5), 1.0);
  }
  void main() {
    float fade = 1.0 - smoothstep(15.0, 60.0, distance(cameraPosition, vWorld));
    float minor = grid(0.1);
    float major = grid(1.0);
    float ax = axisLine(vLocal.y); // local x axis (world x)
    float az = axisLine(vLocal.x); // local -y axis (world z)
    vec3 color = mix(vec3(0.42, 0.45, 0.5), vec3(0.62, 0.65, 0.7), major);
    color = mix(color, vec3(0.85, 0.3, 0.28), ax);
    color = mix(color, vec3(0.3, 0.55, 0.9), az);
    float alpha = max(max(minor * 0.35, major * 0.75), max(ax, az)) * fade;
    gl_FragColor = vec4(color, alpha);
  }`;

export function rebuild(scene) {
  root.traverse(obj => {
    obj.geometry?.dispose();
    obj.material?.dispose();
  });
  root.clear();
  bodyGroups.length = 0;
  pickables.length = 0;
  for (const map of Object.values(objectsById)) map.clear();
  if (!scene) return;

  // ---------- bodies: group at the body pose holding the mesh + edge lines ----------
  for (const body of scene.bodies) {
    const group = new THREE.Group();
    group.position.set(...body.position);
    const [qw, qx, qy, qz] = body.rotation;
    group.quaternion.set(qx, qy, qz, qw);
    group.userData.isPlane = body.shape.type === 'plane';
    root.add(group);
    bodyGroups.push(group);
    objectsById.body.set(body.id, group);
    const shape = body.shape;

    if (shape.type === 'plane') {
      const ground = new THREE.Mesh(
        new THREE.PlaneGeometry(400, 400),
        new THREE.MeshStandardMaterial({ color: body.color, roughness: 1, metalness: 0 }),
      );
      // PlaneGeometry lies in local xy; rotate so its normal is local +y like the spec's plane
      ground.rotation.x = -Math.PI / 2;
      ground.receiveShadow = true;
      const grid = new THREE.Mesh(
        ground.geometry,
        new THREE.ShaderMaterial({
          vertexShader: GRID_VERTEX,
          fragmentShader: GRID_FRAGMENT,
          transparent: true,
          depthWrite: false,
          polygonOffset: true,
          polygonOffsetFactor: -1,
          polygonOffsetUnits: -1,
        }),
      );
      grid.rotation.x = -Math.PI / 2;
      group.add(ground, grid);
      group.userData.setSelected = (on) => ground.material.emissive.set(on ? 0x0d2747 : 0x000000);
      continue;
    }

    let geometry;
    if (shape.type === 'box') geometry = new THREE.BoxGeometry(...shape.size);
    else if (shape.type === 'sphere') geometry = new THREE.SphereGeometry(shape.radius, 48, 24);
    else if (shape.type === 'cylinder') geometry = new THREE.CylinderGeometry(shape.radius, shape.radius, shape.height, 48);
    else throw new Error(`unknown shape type "${shape.type}" on body "${body.id}"`);

    const material = new THREE.MeshStandardMaterial({
      color: body.color,
      roughness: 0.55,
      metalness: 0.05,
      // non-colliding bodies (e.g. target zones) are drawn see-through
      transparent: !body.collide,
      opacity: body.collide ? 1 : 0.75,
      depthWrite: body.collide,
      // push faces back a little so the edge lines don't z-fight
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = body.collide;
    mesh.receiveShadow = true;
    mesh.userData = { kind: 'body', id: body.id };
    pickables.push(mesh);

    // edge lines that make the orientation visible: box edges, sphere great circles, cylinder caps + spokes + side lines
    let edgeGeometry;
    if (shape.type === 'box') {
      edgeGeometry = new THREE.EdgesGeometry(geometry);
    } else {
      const points = []; // [x, y, z, x, y, z, ...] pairs of segment endpoints
      const N = 64;
      // closed loop through point(angle)
      const ring = (point) => {
        for (let i = 0; i < N; i++) points.push(...point((i / N) * 2 * Math.PI), ...point(((i + 1) / N) * 2 * Math.PI));
      };
      const r = shape.radius;
      if (shape.type === 'sphere') {
        ring(a => [r * Math.cos(a), r * Math.sin(a), 0]);
        ring(a => [r * Math.cos(a), 0, r * Math.sin(a)]);
        ring(a => [0, r * Math.cos(a), r * Math.sin(a)]);
      } else {
        const h = shape.height / 2;
        for (const y of [-h, h]) {
          ring(a => [r * Math.cos(a), y, r * Math.sin(a)]);
          points.push(-r, y, 0, r, y, 0, 0, y, -r, 0, y, r);
        }
        for (const a of [0, 0.5 * Math.PI, Math.PI, 1.5 * Math.PI]) {
          points.push(r * Math.cos(a), -h, r * Math.sin(a), r * Math.cos(a), h, r * Math.sin(a));
        }
      }
      edgeGeometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    }
    const edgeMaterial = new THREE.LineBasicMaterial({ color: EDGE_COLOR, transparent: true, opacity: 0.7 });
    group.add(mesh, new THREE.LineSegments(edgeGeometry, edgeMaterial));
    group.userData.setSelected = (on) => {
      material.emissive.set(on ? 0x0d2747 : 0x000000);
      edgeMaterial.color.set(on ? SELECT_COLOR : EDGE_COLOR);
      edgeMaterial.opacity = on ? 1 : 0.7;
    };
  }

  // ---------- joints: anchor sphere + axis arrow, attached to body_b so it moves with it during simulation ----------
  for (const joint of scene.joints) {
    findItem(scene, 'body', joint.body_a);
    const bodyB = findItem(scene, 'body', joint.body_b);
    const color = JOINT_COLORS[joint.type];
    if (color === undefined) throw new Error(`unknown joint type "${joint.type}" on joint "${joint.id}"`);

    // anchor / axis are world-space at definition time -> convert to body_b's local frame
    const [qw, qx, qy, qz] = bodyB.rotation;
    const invB = new THREE.Quaternion(qx, qy, qz, qw).invert();
    const localAnchor = new THREE.Vector3(...joint.anchor).sub(new THREE.Vector3(...bodyB.position)).applyQuaternion(invB);
    const localAxis = new THREE.Vector3(...joint.axis).normalize().applyQuaternion(invB);

    const marker = new THREE.Group();
    marker.position.copy(localAnchor);
    // drawn on top of bodies so joints inside bodies stay visible
    const sphereMaterial = new THREE.MeshBasicMaterial({ color, depthTest: false });
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 8), sphereMaterial);
    sphere.renderOrder = 10;
    sphere.userData = { kind: 'joint', id: joint.id };
    marker.add(sphere);
    pickables.push(sphere);
    if (joint.type !== 'fixed') {
      const arrow = new THREE.ArrowHelper(localAxis, new THREE.Vector3(), 0.1, color, 0.025, 0.014);
      arrow.line.material.depthTest = false;
      arrow.cone.material.depthTest = false;
      arrow.line.renderOrder = 10;
      arrow.cone.renderOrder = 10;
      marker.add(arrow);
    }
    marker.userData.setSelected = (on) => {
      sphereMaterial.color.set(on ? 0xffffff : color);
      marker.scale.setScalar(on ? 1.4 : 1);
    };
    objectsById.body.get(joint.body_b).add(marker);
    objectsById.joint.set(joint.id, marker);
  }

  // ---------- cameras: small frustum (apex + image rectangle + "up" triangle) in the body frame, looking down -z ----------
  for (const cam of scene.cameras) {
    findItem(scene, 'body', cam.body);
    const depth = 0.06;
    const hh = depth * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)); // half height of the image rectangle
    const hw = hh * (cam.width / cam.height); // half width
    const z = -depth;
    const corners = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]; // [[x, y]] image rectangle corners
    const points = []; // [x, y, z, ...] segment endpoint pairs
    corners.forEach(([x, y], i) => {
      const [nx, ny] = corners[(i + 1) % 4];
      points.push(0, 0, 0, x, y, z, x, y, z, nx, ny, z);
    });
    points.push(-hw * 0.5, hh * 1.1, z, 0, hh * 1.5, z, 0, hh * 1.5, z, hw * 0.5, hh * 1.1, z, hw * 0.5, hh * 1.1, z, -hw * 0.5, hh * 1.1, z);
    const material = new THREE.LineBasicMaterial({ color: CAMERA_COLOR, depthTest: false });
    const lines = new THREE.LineSegments(
      new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(points, 3)),
      material,
    );
    lines.renderOrder = 10;
    lines.position.set(...cam.position);
    const [qw, qx, qy, qz] = cam.rotation;
    lines.quaternion.set(qx, qy, qz, qw);
    lines.userData = { kind: 'camera', id: cam.id };
    lines.userData.setSelected = (on) => material.color.set(on ? 0xffffff : CAMERA_COLOR);
    pickables.push(lines);
    objectsById.body.get(cam.body).add(lines);
    objectsById.camera.set(cam.id, lines);
  }
  highlight();
}

export function highlight() {
  const selected = new Set(app.selection.map(s => `${s.kind}:${s.id}`)); // {"kind:id"}
  for (const [kind, map] of Object.entries(objectsById)) {
    for (const [id, obj] of map) obj.userData.setSelected(selected.has(`${kind}:${id}`));
  }
}

// flat = 7 * numBodies floats: per body (px, py, pz, qw, qx, qy, qz) in scene["bodies"] order
export function applyPoses(flat) {
  if (flat.length !== bodyGroups.length * 7) {
    throw new Error(`got ${flat.length / 7} poses for ${bodyGroups.length} bodies`);
  }
  bodyGroups.forEach((group, i) => {
    const o = i * 7;
    group.position.set(flat[o], flat[o + 1], flat[o + 2]);
    group.quaternion.set(flat[o + 4], flat[o + 5], flat[o + 6], flat[o + 3]);
  });
}
