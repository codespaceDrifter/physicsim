# Physics Simulation - How It Works

This document explains the entire codebase from the ground up. Read it top to bottom and you'll understand everything.

---

## Part 1: The Object

Everything in this simulation starts with an object. An object is a physical thing - a planet, a ball, a table. In code, this is `Obj` in `back/obj.py`.

An object has two parts: its **shape** (what it looks like) and its **physics state** (where it is, how fast it's moving, etc).

### The Mesh

Before you can have a physics object, you need a shape. A shape is defined by a **mesh** - a collection of triangles that approximate a 3D surface. The mesh lives in `back/mesh.py`.

A mesh has:
- **vertices**: a list of 3D points `(x, y, z)`
- **indices**: which vertices connect to form triangles (every 3 indices = 1 triangle)
- **volume**: how much space the mesh encloses (used to calculate mass)
- **AABB**: axis-aligned bounding box (the smallest box that contains the mesh, used for fast collision checks)

You don't create meshes manually. Instead, `back/shapes.py` has helper functions:

```python
mesh = sphere(radius=10, subdivisions=2)
mesh = box(w=10, h=20, d=30)
```

The sphere function creates an icosahedron and subdivides it to approximate a sphere. The box function just makes 8 vertices and 12 triangles. `hollow_rect_frame()` builds the billiard table rails out of 4 boxes.

The mesh is re-centered so its volume-weighted center of mass sits at the origin.

### The Physics State

Now that we have a shape, we wrap it in an `Obj`. This adds the physics (linear only, no rotation yet):

- `position` - where the center of mass is in world space
- `velocity` - how fast it's moving (meters per second)
- `force` - forces currently acting on it (gets reset each frame)
- `mass` - calculated from mesh volume × density
- `inv_mass` - 1/mass, or 0 for `static=True` bodies (infinite mass, never move)
- `id`, `color`, `restitution` (1.0 = perfectly elastic, 0 = no bounce)

Creating an object looks like this:

```python
obj = Obj(
    'earth', '#3fa9c0',
    sphere(40, 2),
    5000.0,                                       # density
    torch.tensor([0., 0., 0.], device=config.device),   # position
    torch.tensor([0., 0., -5.], device=config.device),  # velocity
)
```

---

## Part 2: The World

Objects don't exist in isolation. They live in a **World** (`back/world.py`). The world is the container that holds all objects and runs the simulation.

```python
world = World('mySimulation', time_step=0.01)
world.add_object(obj1)
world.add_object(obj2)
```

The `time_step` is how much time passes each simulation tick. Smaller = more accurate but slower.

### The Simulation Loop

Every frame, the world calls `update()`. This is where physics happens. The order matters:

**Step 1: Detect collisions**

The world checks every pair of objects to see if they're touching. This happens in `back/collision.py`.

First, a fast check: do their bounding boxes overlap? If not, they can't possibly be touching, skip them. This is the "broad phase".

If boxes overlap, we do the expensive check: does any triangle edge from object A pass through any triangle of object B? This uses the Möller-Trumbore algorithm, fully parallelized on GPU. All edge-triangle pairs are tested simultaneously.

**Step 2: Resolve collisions**

If two objects are colliding, we need to bounce them apart. This happens in `back/mechanics.py` in `apply_collision_response()`.

First, we step both (non-static) objects backwards in time slightly (undo the penetration). Then we calculate an impulse along the line between their centers - an instantaneous change in velocity that makes them bounce. The impulse depends on their relative velocity, their inverse masses, and their restitution.

Known limitation: the center-to-center normal is only correct for spheres. Ball vs table rail uses the wrong normal.

**Step 3: Apply gravity**

Every object pulls on every other object via gravity. This is `apply_gravity_batched()` in `back/mechanics.py`.

The formula is Newton's law: `F = G * m1 * m2 / r²`

With N objects, there are N² force pairs. We compute all of them in parallel on GPU:
- Stack all positions into a (N, 3) tensor
- Compute pairwise distances: (N, N) tensor
- Compute force magnitudes: (N, N) tensor
- Compute force vectors: (N, N, 3) tensor
- Sum forces for each object: (N, 3) tensor

Each object's `force` attribute gets updated.

**Step 4: Integrate motion**

Finally, each object updates its position based on accumulated forces. This is `Obj.update()`.

Semi-implicit Euler:
```
acceleration = force * inv_mass
velocity = velocity + acceleration * dt
position = position + velocity * dt
```

After integration, forces are reset to zero for the next frame.

---

## Part 3: Recording

Running a simulation is slow. We don't want to simulate in real-time while the user watches. Instead, we simulate once, save the results, and play them back later.

The `Recording` class in `back/recording.py` handles this.

```python
recording = Recording(
    name='threeBody',
    world=world,
    seconds_per_frame=0.5,  # save a frame every 0.5 seconds
    end_seconds=3000        # simulate for 3000 seconds total
)
recording.simulate()
```

This runs the simulation loop, and periodically saves "snapshots" of the world state. Each snapshot contains, for every object:
- ID
- Position (3 floats)
- Vertices (the mesh geometry)
- Indices (triangle connectivity)
- Color

### The Binary Format

Snapshots are saved to `logs/{name}/{chunk}.bin` in a custom binary format. We use binary because it's compact and fast to parse.

Each file (chunk) contains up to 1000 frames. The format is:

```
Header:
  "SIMB"           (4 bytes magic)
  seconds_per_frame (float32)
  total_frames      (uint32)
  frames_in_chunk   (uint32)

For each frame:
  time              (float32)
  object_count      (uint32)

  For each object:
    id_length       (uint32)
    id              (bytes)
    position        (3 × float32)
    vertex_count    (uint32)
    vertices        (N × float32)
    index_count     (uint32)
    indices         (N × uint32)
    color_length    (uint32)
    color           (bytes)
```

All numbers are little-endian.

---

## Part 4: The Server

The backend runs a web server (`back/api.py`) using FastAPI. It serves:

- `GET /recordings/{name}/{chunk}` - returns a binary chunk file
- `GET /worlds` - returns list of available recordings
- `GET /` - serves the frontend HTML

```bash
cd back
python -m api  # runs on http://localhost:8080
```

---

## Part 5: The Frontend

The frontend is a Three.js app that plays back recordings. It lives in `front/`.

### State Management

All state lives in `front/src/state.js`:

```javascript
playback = {
  world: 'threeBody',  // which recording to play
  frame: 0,            // current frame number
  chunk: 0,            // current chunk number
  speed: 1,            // playback speed multiplier
  maxFrame: 0,         // total frames in recording
}

cache = {
  frames: [],          // decoded frames from current chunk
  relativeFrame: 0,    // index within current chunk
}
```

### Fetching and Decoding

`front/src/api.js` fetches binary chunks from the server and decodes them.

```javascript
const recording = await fetchRecording('threeBody', 0);
// recording.frames is an object: { "0.000": frameData, "0.500": frameData, ... }
```

The decoder reads the binary format byte-by-byte, extracting all the fields. The output matches the backend structure but with JavaScript naming (camelCase).

### Playback Loop

`front/src/playback.js` controls frame advancement.

The `step()` function runs every 33ms (30 FPS):

1. Check if we've reached the end → reset to beginning
2. Check if current chunk is exhausted → load next chunk
3. Get current frame data from cache
4. Call `updateSimulation(frameData)` to render it
5. Advance frame counter
6. Schedule next step via setTimeout

The `reset()` function loads chunk 0 and resets all counters.

### Rendering Objects

`front/src/objects.js` manages Three.js meshes.

It keeps a Map of object ID → Three.js Mesh. Each frame:

1. For each object in the frame data:
   - If we already have a mesh for this ID, update its position
   - If not, create a new mesh from the vertices/indices/color
2. For each mesh we have:
   - If the object is no longer in the frame, remove it from the scene

Creating a mesh:
```javascript
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
geometry.setIndex(indices);

const material = new THREE.MeshStandardMaterial({ color: color });
const mesh = new THREE.Mesh(geometry, material);
mesh.position.set(x, y, z);
scene.add(mesh);
```

### Scene Setup

`front/src/scene.js` creates the Three.js environment:

- A scene with black background
- Ambient light (so objects aren't in shadow)
- Directional light (for depth/shading)
- A perspective camera starting at (100, 100, 100)
- OrbitControls for mouse interaction (rotate, pan, zoom)
- An axes helper showing X/Y/Z directions

### User Controls

`front/src/controls.js` handles keyboard input:

- Left/Right arrow: decrease/increase playback speed
- R: reset to beginning
- Mouse: handled by OrbitControls (left-drag rotate, right-drag pan, scroll zoom)

It also creates the world selection buttons in the UI.

### Entry Point

`front/src/main.js` ties it together:

```javascript
import { scene, camera, renderer, controls } from './scene.js';
import { step } from './playback.js';
import './controls.js';

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

animate();  // start render loop
step();     // start playback
```

Two loops run independently:
- `animate()` runs as fast as possible (60 FPS typically), rendering the current state
- `step()` runs at 30 FPS, advancing the simulation frame

---

## Part 6: Example Simulations

### Three Body (`back/example_worlds/three_body.py`)

Simulates a three-star system inspired by the Three Body Problem:
- Alpha Centauri A (yellow, large)
- Alpha Centauri B (orange, medium)
- Proxima Centauri (red, large)
- Trisolaris (small planet orbiting)

They all gravitationally attract each other, creating chaotic orbital dynamics.

### Billiard Ball (`back/example_worlds/billard_ball.py`)

Simulates billiard balls on a table:
- A hollow rectangular frame (the table edges)
- Multiple rows of balls, some stationary, some moving

Balls collide with each other and bounce off the table frame.

---

## Running Everything

**Generate recordings:**
```bash
cd back
python -m simulate
```

**Start the server:**
```bash
python -m api
```

**Run the frontend (development):**
```bash
cd front
npm run dev
```

Open http://localhost:5173 and watch the simulation play.

---

## File Summary

```
back/
  mesh.py           - Mesh class (vertices, indices, volume, AABB)
  shapes.py         - sphere(), box() mesh generators
  obj.py            - Obj class (position, velocity, mass, static)
  config.py         - torch device selection
  world.py          - World class (holds objects, runs simulation loop)
  collision.py      - Collision detection (AABB + triangle intersection)
  mechanics.py      - Forces (gravity) and collision response (linear impulses)
  recording.py      - Recording class (runs simulation, saves binary files)
  api.py            - FastAPI server
  simulate.py       - Runs all example simulations
  example_worlds/   - Pre-built simulation setups

front/src/
  state.js          - Global state (playback position, cache)
  api.js            - Fetch and decode binary recordings
  playback.js       - Frame-by-frame playback control
  objects.js        - Create/update/remove Three.js meshes
  scene.js          - Three.js scene, camera, lights, controls
  controls.js       - Keyboard input, world selection UI
  main.js           - Entry point, starts render and playback loops

logs/
  {worldName}/      - Binary recording files (0.bin, 1.bin, ...)
```
