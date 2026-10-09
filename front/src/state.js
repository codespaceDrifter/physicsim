// Global app state + a tiny event bus. Every module reads the scene JSON from here.

export const app = {
  // edited scene JSON, same format as disk / backend (single source of truth)
  scene: null,
  // scene JSON currently displayed: app.scene in edit/simulate, the recording's scene in playback
  shown: null,
  // 'edit' | 'simulate' | 'playback'
  mode: 'edit',
  // last item is the primary selection
  selection: [], // [{kind: 'body' | 'joint' | 'camera', id: str}]
};

// events:
//   'scene' (source)  app.shown changed (structure or values); source = who changed it
//   'selection'       app.selection changed
//   'pose'            gizmo is dragging a body (body pose in app.scene changed)
//   'mode' (prev)     app.mode changed
//   'camera'          viewport switched between perspective / orthographic camera
//   'simFrame' (msg), 'simStatus' (msg), 'simPlay'   websocket simulation events
const listeners = {}; // {eventName: [fn]}

export function on(name, fn) {
  (listeners[name] ??= []).push(fn);
}

export function emit(name, arg) {
  for (const fn of listeners[name] ?? []) fn(arg);
}

export function select(items) {
  app.selection = items;
  emit('selection');
}

// click selection used by viewport + outliner: shift toggles bodies in a multi-selection
export function clickSelect(item, shift) {
  if (!shift || !item || item.kind !== 'body') {
    select(item ? [item] : []);
    return;
  }
  const bodies = app.selection.filter(s => s.kind === 'body'); // [{kind, id}]
  const already = bodies.some(s => s.id === item.id);
  select(already ? bodies.filter(s => s.id !== item.id) : [...bodies, item]);
}

export function setMode(mode) {
  if (mode === app.mode) return;
  const prev = app.mode;
  app.mode = mode;
  emit('mode', prev);
}
