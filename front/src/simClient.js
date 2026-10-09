// Live simulation over the /ws websocket (protocol: spec section 4).
import { app, on, emit } from './state.js';
import { applyPoses } from './sceneView.js';
import { toast } from './toast.js';

export const sim = {
  state: 'stopped', // 'running' | 'paused' | 'stopped', as last reported by the server
  autopilot: 'off',
  time: 0, // sim time of the last frame, seconds
};

let ws = null; // WebSocket, null until the first play

export function send(msg) {
  if (ws === null || ws.readyState !== WebSocket.OPEN) throw new Error(`simulation not connected, cannot send "${msg.type}"`);
  ws.send(JSON.stringify(msg));
}

export async function play() {
  if (ws === null) {
    await new Promise((resolve, reject) => {
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${protocol}://${location.host}/ws`);
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error('could not connect to simulation websocket /ws'));
      ws.onclose = () => {
        ws = null;
        if (sim.state !== 'stopped') toast('simulation connection closed');
        sim.state = 'stopped';
        emit('simStatus');
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'frame') {
          // frames that arrive after leaving simulate mode (before the server processed "stop") are dropped
          if (app.mode !== 'simulate') return;
          applyPoses(msg.poses.flat());
          sim.time = msg.time;
          emit('simFrame', msg);
        } else if (msg.type === 'status') {
          sim.state = msg.state;
          sim.autopilot = msg.autopilot;
          emit('simStatus');
        } else if (msg.type === 'error') {
          toast(`simulation: ${msg.message}`);
        } else {
          throw new Error(`unknown websocket message type "${msg.type}"`);
        }
      };
    });
  }
  sim.time = 0;
  emit('simPlay');
  send({ type: 'play', scene: app.scene });
}

// leaving simulate mode discards the world
on('mode', (prev) => {
  if (prev !== 'simulate' || sim.state === 'stopped') return;
  send({ type: 'stop' });
  sim.state = 'stopped';
  emit('simStatus');
});
