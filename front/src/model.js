// Scene JSON helpers shared by editor, inspector, outliner, toolbar (schema: spec section 1)

export const LIST_KEY = { body: 'bodies', joint: 'joints', camera: 'cameras' }; // {kind: scene key}

// default shapes for new bodies / shape type changes, all 0.2 m tall
export const DEFAULT_SHAPES = {
  box: { type: 'box', size: [0.2, 0.2, 0.2] },
  sphere: { type: 'sphere', radius: 0.1 },
  cylinder: { type: 'cylinder', radius: 0.1, height: 0.2 },
  plane: { type: 'plane' },
}; // {shape type: shape}

export function findItem(scene, kind, id) {
  const item = scene[LIST_KEY[kind]].find(x => x.id === id);
  if (!item) throw new Error(`scene "${scene.name}" has no ${kind} "${id}"`);
  return item;
}

// base if free in list, else base_2, base_3, ...
export function uniqueId(list, base) {
  const taken = new Set(list.map(x => x.id)); // {id}
  if (!taken.has(base)) return base;
  let n = 2;
  while (taken.has(`${base}_${n}`)) n++;
  return `${base}_${n}`;
}

export function makeBody(id, shape, position) {
  return {
    id,
    shape,
    position,
    rotation: [1, 0, 0, 0],
    velocity: [0, 0, 0],
    angular_velocity: [0, 0, 0],
    density: 1000.0,
    static: shape.type === 'plane',
    collide: true,
    group: 0,
    friction: 0.6,
    restitution: 0.0,
    color: shape.type === 'plane' ? '#3b3f46' : '#9fb3c8',
  };
}

export function newScene(name) {
  return {
    name,
    settings: { dt: 1 / 60, substeps: 4, gravity: [0, -9.81, 0], G: 0.0 },
    bodies: [makeBody('ground', { type: 'plane' }, [0, 0, 0])],
    joints: [],
    cameras: [],
  };
}
