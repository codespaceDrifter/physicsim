/**
 * Object management module.
 * Handles creation, update, and removal of Three.js meshes from simulation data.
 *
 * Frame data format (from api.js):
 * {
 *   curTime: number,
 *   idArray: string[],
 *   positionArrays: number[][],      // [x, y, z] per object
 *   vertexArrays: number[][],        // flat vertex positions per object
 *   indexArrays: number[][],         // triangle indices per object
 *   colorArray: string[],            // hex color per object
 * }
 */
import * as THREE from 'three';
import { scene } from './scene.js';

// Map of object ID -> Three.js Mesh
const objects = new Map();

export function updateSimulation(frameData) {
  const {
    curTime,
    idArray,
    positionArrays,
    vertexArrays,
    indexArrays,
    colorArray,
  } = frameData;

  const elapsedTimeDiv = document.getElementById('elapsed-time');
  if (elapsedTimeDiv && typeof curTime === 'number') {
    elapsedTimeDiv.textContent = `Time: ${curTime.toFixed(2)}s`;
  }

  idArray.forEach((id, index) => {
    const objectData = {
      position: positionArrays[index],
      vertices: vertexArrays[index],
      indices: indexArrays[index],
      color: colorArray[index],
    };

    if (objects.has(id)) {
      updateObject(id, objectData);
    } else {
      createObject(id, objectData);
    }
  });

  for (const id of objects.keys()) {
    if (!idArray.includes(id)) {
      removeObject(id);
    }
  }
}

function createObject(id, data) {
  const { position, vertices, indices, color } = data;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(new Float32Array(vertices), 3)
  );
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));

  const material = new THREE.MeshStandardMaterial({
    color: color ? new THREE.Color(color) : new THREE.Color(0xffffff),
    roughness: 0.7,
    metalness: 0.3,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(position[0], position[1], position[2]);

  const edges = new THREE.EdgesGeometry(geometry);
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
  const wireframe = new THREE.LineSegments(edges, lineMaterial);
  mesh.add(wireframe);
  mesh.userData.wireframe = wireframe;

  scene.add(mesh);
  objects.set(id, mesh);
}

function updateObject(id, data) {
  const mesh = objects.get(id);
  if (!mesh) return;
  mesh.position.set(data.position[0], data.position[1], data.position[2]);
}

function removeObject(id) {
  const mesh = objects.get(id);
  if (mesh) {
    scene.remove(mesh);
    objects.delete(id);
  }
}

export function clearObjects() {
  for (const [id, mesh] of objects.entries()) {
    scene.remove(mesh);
  }
  objects.clear();
}
