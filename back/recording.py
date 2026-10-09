# Recordings for the frontend's playback mode: the scene JSON once, then only body poses per frame.
#
# logs/{name}/scene.json   the scene the recording was made from (bodies define shapes/colors)
# logs/{name}/{chunk}.bin  little-endian:
#   magic "SIMR", seconds_per_frame float32, total_frames uint32, frames_in_chunk uint32, num_bodies uint32
#   per frame: time float32, then num_bodies * 7 float32 (px, py, pz, qw, qx, qy, qz) in scene body order
import json
import struct
from pathlib import Path
import numpy as np

LOGS_DIR = Path(__file__).parent.parent / 'logs'
FRAMES_PER_CHUNK = 1000


class Recorder:
    def __init__(self, name, scene, seconds_per_frame, total_frames):
        self.dir = LOGS_DIR / name
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / 'scene.json', 'w') as f:
            json.dump(scene, f)
        self.seconds_per_frame = seconds_per_frame
        self.total_frames = total_frames
        self.num_bodies = len(scene['bodies'])
        # [(time, (N, 7) float32 poses)] frames of the chunk being filled
        self.frames = []
        self.chunk = 0
        self.written = 0

    def add(self, time, poses):
        # poses: (N, 7) array-like for one env
        poses = np.asarray(poses, dtype=np.float32)
        assert poses.shape == (self.num_bodies, 7)
        assert self.written + len(self.frames) < self.total_frames, 'more frames than total_frames'
        self.frames.append((time, poses))
        if len(self.frames) == FRAMES_PER_CHUNK or self.written + len(self.frames) == self.total_frames:
            with open(self.dir / f'{self.chunk}.bin', 'wb') as f:
                f.write(b'SIMR')
                f.write(struct.pack('<fIII', self.seconds_per_frame, self.total_frames, len(self.frames), self.num_bodies))
                for t, p in self.frames:
                    f.write(struct.pack('<f', t))
                    f.write(p.astype('<f4').tobytes())
            self.written += len(self.frames)
            self.frames = []
            self.chunk += 1


def record_world(name, world, seconds_per_frame, end_seconds):
    # step env 0 of a world for end_seconds, saving a frame every seconds_per_frame
    total = int(round(end_seconds / seconds_per_frame)) + 1
    rec = Recorder(name, world.scene, seconds_per_frame, total)
    for i in range(total):
        target = i * seconds_per_frame
        while world.time < target - 1e-9:
            world.step()
        rec.add(target, world.poses()[0].cpu().numpy())
        if i % 200 == 0:
            print(f'{name}: frame {i}/{total}')
