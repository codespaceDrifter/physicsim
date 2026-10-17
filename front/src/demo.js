// Simple view: plays the pre-recorded games (back/record_games.py, recorded on a GPU) back to back. This web server
// has no GPU, so it only serves the recordings. Shows the score, a countdown and the robots' 4 camera feeds.
import { app, on, emit, setMode, setView } from './state.js';
import { getJson, getBinary } from './api.js';
import { applyPoses, showHelpers } from './sceneView.js';
import { camera, controls, onTick } from './viewport.js';

const playBtn = document.getElementById('demo-play');
const scoreEl = document.getElementById('demo-score');
const clockEl = document.getElementById('demo-clock');
const feedsEl = document.getElementById('demo-feeds');
const driverEls = ['orange', 'teal'].map(c => document.getElementById(`demo-${c}`)); // the scoreboard's driver captions
const viewButtons = document.querySelectorAll('.views button');
const HEADER_BYTES = 20; // recording chunk header: magic, seconds_per_frame, total_frames, frames_in_chunk, num_bodies
const FRAMES_PER_CHUNK = 1000;
// seconds the final score stays up before the next game
const RESULT_SECONDS = 4;
// fixed view of the arena (2.4 x 1.6 m): from the south, above, aimed a bit right so the camera feeds don't cover it
const VIEW_POSITION = [0.25, 1.7, 1.65];
const VIEW_TARGET = [0.25, 0, 0.05];
// scoreboard caption per driver (game.json "drivers")
const DRIVER_LABELS = { // {driver: caption}
  teacher: 'teacher model · sees exact positions',
  student: 'student model · sees only its cameras',
  expert: 'scripted expert',
};
// camera feed titles: robot prefix -> its color name
const ROBOT_NAMES = { 'lekiwi/': 'orange', 'lekiwi2/': 'teal' }; // {prefix: name}

const video = document.createElement('video'); // the current game's camera mosaic, drawn into the 4 feeds
video.muted = true;
video.playsInline = true;
const demo = {
  games: [], // [recording name]
  index: -1, // game being shown
  game: null, // its game.json {stage, seconds, drivers, cameras, score: [[time, orange, teal]]} + scene, frames
  t: 0, // playhead, s (runs past game.seconds while the result shows)
  playing: true,
  feeds: [], // [{ctx, x, y}] canvas + its tile in the mosaic, "cameras" order
};

async function loadGame(index) {
  const name = demo.games[index];
  const [scene, game] = await Promise.all([getJson(`/recordings/${name}/scene`), getJson(`/recordings/${name}/game`)]);
  // every chunk up front (a 60 s game is ~2 MB): [Float32Array] per chunk, per frame time then poses
  const first = new DataView(await getBinary(`/recordings/${name}/0`));
  if (String.fromCharCode(...[0, 1, 2, 3].map(i => first.getUint8(i))) !== 'SIMR') throw new Error(`${name}: not a recording`);
  const total = first.getUint32(8, true);
  const chunks = [first.buffer];
  for (let c = 1; c * FRAMES_PER_CHUNK < total; c++) chunks.push(await getBinary(`/recordings/${name}/${c}`));
  const stride = 1 + 7 * scene.bodies.length; // floats per frame
  game.frames = chunks.map(b => new Float32Array(b, HEADER_BYTES, new DataView(b).getUint32(12, true) * stride));
  game.secondsPerFrame = first.getFloat32(4, true);
  game.total = total;
  game.stride = stride;
  game.scene = scene;
  demo.index = index;
  demo.game = game;
  demo.t = 0;
  video.src = `/recordings/${name}/cameras`;
  if (demo.playing) video.play();
  makeFeeds(game.cameras);
  game.drivers.forEach((d, i) => { driverEls[i].textContent = DRIVER_LABELS[d]; });
  if (app.view === 'simple') show();
}

function show() {
  // put the game's scene on screen, seen from the fixed view
  app.shown = demo.game.scene;
  emit('scene', 'demo');
  camera.position.set(...VIEW_POSITION);
  controls.target.set(...VIEW_TARGET);
  controls.update();
  draw();
}

function makeFeeds(ids) {
  feedsEl.innerHTML = '';
  demo.feeds = ids.map((id, i) => {
    const panel = document.createElement('div');
    panel.className = 'feed';
    panel.innerHTML = '<div class="feed-title"></div><canvas></canvas>';
    const slash = id.lastIndexOf('/') + 1;
    panel.firstChild.textContent = `${ROBOT_NAMES[id.slice(0, slash)]} robot · ${id.slice(slash).replace('_', ' ')}`;
    feedsEl.append(panel);
    return { ctx: panel.lastChild.getContext('2d'), x: i % 2, y: Math.floor(i / 2) };
  });
}

function draw() {
  // the frame under the playhead, the score at that time, the clock and the camera feeds
  const g = demo.game;
  const frame = Math.min(Math.floor(demo.t / g.secondsPerFrame), g.total - 1);
  const data = g.frames[Math.floor(frame / FRAMES_PER_CHUNK)];
  const offset = (frame % FRAMES_PER_CHUNK) * g.stride;
  applyPoses(data.subarray(offset + 1, offset + g.stride));
  const [, orange, teal] = g.score.findLast(([time]) => time <= demo.t);
  scoreEl.textContent = `${orange} : ${teal}`;
  const left = Math.max(0, Math.ceil(g.seconds - demo.t));
  const result = orange === teal ? 'draw' : `${orange > teal ? 'orange' : 'teal'} wins`;
  const where = `game ${demo.index + 1} of ${demo.games.length}`;
  clockEl.innerHTML = demo.t < g.seconds
    ? `<b>0:${String(left).padStart(2, '0')}</b> left · ${where}`
    : `<b>${result}</b> · next game in ${Math.ceil(g.seconds + RESULT_SECONDS - demo.t)} s`;
  if (video.videoWidth === 0) return;
  // each camera is one tile of the 2 x 2 mosaic
  const w = video.videoWidth / 2;
  const h = video.videoHeight / 2;
  for (const { ctx, x, y } of demo.feeds) {
    if (ctx.canvas.width !== w) {
      ctx.canvas.width = w;
      ctx.canvas.height = h;
    }
    ctx.drawImage(video, x * w, y * h, w, h, 0, 0, w, h);
  }
}

onTick((dt) => {
  if (app.view !== 'simple' || demo.game === null || !demo.playing) return;
  demo.t += dt;
  if (demo.t >= demo.game.seconds + RESULT_SECONDS) {
    // no game while the next one loads, so this tick doesn't fire again
    demo.game = null;
    loadGame((demo.index + 1) % demo.games.length);
    return;
  }
  // keep the camera video on the game clock (it stops at its end, while the result shows)
  if (demo.t < demo.game.seconds && Math.abs(video.currentTime - demo.t) > 0.3) video.currentTime = demo.t;
  draw();
});

function setPlaying(playing) {
  demo.playing = playing;
  playBtn.classList.toggle('on', playing);
  playBtn.textContent = `${playing ? '❚❚ Playing' : '▶ Play'} pre-recorded games · this web server has no GPU`;
  if (playing && demo.game !== null && demo.t < demo.game.seconds) video.play();
  else video.pause();
}

playBtn.onclick = () => setPlaying(!demo.playing);

export async function startGames() {
  demo.games = (await getJson('/games')).games;
  if (demo.games.length === 0) throw new Error('no pre-recorded games on the server (back/record_games.py)');
  await loadGame(0);
}

for (const btn of viewButtons) btn.onclick = () => setView(btn.dataset.view);

on('view', () => {
  document.body.dataset.view = app.view;
  for (const btn of viewButtons) btn.classList.toggle('on', btn.dataset.view === app.view);
  showHelpers();
  if (app.view === 'simple') {
    // leave any live simulation / playback, then show the game where it was
    setMode('edit');
    if (demo.game) show();
    setPlaying(true);
    return;
  }
  // advanced: the editor's own scene, the games wait
  setPlaying(false);
  app.shown = app.scene;
  emit('scene', 'view');
});
