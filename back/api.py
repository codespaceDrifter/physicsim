# HTTP + websocket server for the frontend (FastAPI). run from back/:  python -m api   (http://localhost:8080)
#
#   GET  /scenes                      {"scenes": [name]}
#   GET  /scenes/{name}               scene JSON from scenes/{name}.json
#   PUT  /scenes/{name}               save scene JSON
#   GET  /prefabs/lekiwi              {"bodies", "joints", "cameras"} of a LeKiwi at the origin
#   GET  /recordings                  {"recordings": [name]}
#   GET  /recordings/{name}/scene     scene the recording was made from
#   GET  /recordings/{name}/{chunk}   binary chunk (format in recording.py)
#   WS   /ws                          live simulation (protocol in live.py / documentation.md)
import asyncio
import json
import re
import time
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from recording import LOGS_DIR
from robot.lekiwi import lekiwi
from live import LiveSession, CAMERA_EVERY

SCENES_DIR = Path(__file__).parent / 'scenes'
DIST_DIR = Path(__file__).parent.parent / 'front' / 'dist'
# frames sent to the client per second while running
SEND_HZ = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app = FastAPI()


def safe_name(name):
    # scene / recording names become file paths: letters, digits, _ and - only
    if not re.fullmatch(r'[A-Za-z0-9_\-]+', name):
        raise HTTPException(400, f'bad name {name!r}')
    return name


@app.get('/scenes')
def list_scenes():
    return {'scenes': sorted(p.stem for p in SCENES_DIR.glob('*.json'))}


@app.get('/scenes/{name}')
def get_scene(name: str):
    path = SCENES_DIR / f'{safe_name(name)}.json'
    if not path.exists():
        raise HTTPException(404, f'no scene {name}')
    return json.loads(path.read_text())


@app.put('/scenes/{name}')
async def put_scene(name: str, request: Request):
    scene = await request.json()
    SCENES_DIR.mkdir(exist_ok=True)
    (SCENES_DIR / f'{safe_name(name)}.json').write_text(json.dumps(scene, indent=1))
    return {'ok': True}


@app.get('/prefabs/lekiwi')
def prefab_lekiwi():
    return lekiwi()


@app.get('/recordings')
def list_recordings():
    names = sorted(d.name for d in LOGS_DIR.iterdir() if (d / 'scene.json').exists()) if LOGS_DIR.exists() else []
    return {'recordings': names}


@app.get('/recordings/{name}/scene')
def recording_scene(name: str):
    path = LOGS_DIR / safe_name(name) / 'scene.json'
    if not path.exists():
        raise HTTPException(404, f'no recording {name}')
    return FileResponse(path, media_type='application/json')


@app.get('/recordings/{name}/{chunk}')
def recording_chunk(name: str, chunk: int):
    path = LOGS_DIR / safe_name(name) / f'{chunk}.bin'
    if not path.exists():
        raise HTTPException(404, f'no chunk {chunk} in {name}')
    return FileResponse(path, media_type='application/octet-stream')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.websocket('/ws')
async def live(ws: WebSocket):
    # one LiveSession per "play". the sim loop runs as a task next to the message loop and paces itself to real time.
    await ws.accept()
    # mutable connection state: session, running flag, sim task
    state = {'session': None, 'running': False, 'task': None}

    async def send_status():
        session = state['session']
        await ws.send_json({'type': 'status', 'state': 'stopped' if session is None else ('running' if state['running'] else 'paused'),
                            'autopilot': session.autopilot if session else 'off'})

    async def run():
        session = state['session']
        wall_start, sim_start = time.monotonic(), session.world.time
        last_send = 0.0
        while state['running']:
            # step until the sim catches up with the wall clock, then yield to the message loop
            behind = (time.monotonic() - wall_start) - (session.world.time - sim_start)
            if behind > 0.25:
                # can't keep up: drop the backlog instead of spiraling
                wall_start += behind
            if behind > 0:
                session.step()
            now = time.monotonic()
            if now - last_send >= 1 / SEND_HZ:
                last_send = now
                await ws.send_json(session.frame(with_cameras=session.steps % CAMERA_EVERY < 2))
            await asyncio.sleep(0 if behind > 0 else session.world.dt / 2)

    async def stop_task():
        state['running'] = False
        if state['task'] is not None:
            await state['task']
            state['task'] = None

    try:
        while True:
            msg = await ws.receive_json()
            try:
                kind = msg['type']
                if kind == 'play':
                    await stop_task()
                    state['session'] = LiveSession(msg['scene'], DEVICE)
                    state['running'] = True
                    state['task'] = asyncio.create_task(run())
                    await send_status()
                elif kind == 'pause':
                    await stop_task()
                    await send_status()
                elif kind == 'resume':
                    assert state['session'] is not None, 'nothing to resume, press play'
                    if not state['running']:
                        state['running'] = True
                        state['task'] = asyncio.create_task(run())
                    await send_status()
                elif kind == 'step':
                    assert state['session'] is not None, 'nothing to step, press play'
                    await stop_task()
                    state['session'].step()
                    await ws.send_json(state['session'].frame(with_cameras=True))
                    await send_status()
                elif kind == 'stop':
                    await stop_task()
                    state['session'] = None
                    await send_status()
                elif kind == 'drive':
                    if state['session'] is not None:
                        state['session'].drive(msg['targets'])
                elif kind == 'autopilot':
                    assert state['session'] is not None, 'press play first'
                    state['session'].set_autopilot(msg['mode'])
                    await send_status()
                else:
                    raise ValueError(f'unknown message type {kind}')
            except (AssertionError, ValueError, KeyError) as e:
                # bad scene / bad request: tell the user, keep the connection
                await ws.send_json({'type': 'error', 'message': f'{type(e).__name__}: {e}'})
    except WebSocketDisconnect:
        # client gone: the sim task may already have failed sending its last frame, that's expected here
        state['running'] = False
        if state['task'] is not None:
            state['task'].cancel()
            await asyncio.gather(state['task'], return_exceptions=True)


# built frontend (after the API routes so they take precedence)
if DIST_DIR.exists():
    app.mount('/assets', StaticFiles(directory=DIST_DIR / 'assets'), name='assets')

    @app.get('/')
    def index():
        return FileResponse(DIST_DIR / 'index.html')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
