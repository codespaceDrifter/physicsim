# Recordings: one self-contained file per recording, everything a viewer needs (the frontend's playback mode opens it
# from the server or straight from disk): the scene, the visual meshes its bodies show, and every frame's body poses.
#
# logs/{name}.simrec   a gzip stream, little-endian:
#   "SIMREC01"         magic + format version (8 bytes)
#   uint32             header length in bytes
#   header             JSON (utf-8, padded with spaces to a multiple of 4 bytes, so the arrays below stay aligned):
#                        {"scene": the scene JSON (bodies may carry "visuals", see robot/mjcf.py),
#                         "meshes": [{"name": mesh name, "vertices": V, "triangles": T}] in blob order,
#                         "seconds_per_frame": float, "frames": F,
#                         "info": {...} free-form facts about the recording (a demo game's drivers and score)}
#   meshes             per mesh in header order: V x 3 float32 vertex positions (m, in the mesh's own frame), then
#                      T x 3 uint32 vertex indices (counter-clockwise triangles)
#   frames             F x (1 + 7 N) float32: time, then (px, py, pz, qw, qx, qy, qz) per body in scene order
# a mesh name is its file's path under robot/assets/ (binary STL).
import gzip
import json
import struct
from pathlib import Path
import numpy as np

LOGS_DIR = Path(__file__).parent.parent / 'logs'
ASSETS_DIR = Path(__file__).parent / 'robot' / 'assets'
MAGIC = b'SIMREC01'


def load_stl(path):
    # binary STL -> (V, 3) float32 shared vertices, (T, 3) uint32 triangles (STL repeats every corner per triangle;
    # merging equal corners lets the viewer compute smooth normals)
    data = Path(path).read_bytes()
    count = struct.unpack_from('<I', data, 80)[0]
    tri = np.dtype([('normal', '<f4', 3), ('corners', '<f4', (3, 3)), ('attr', '<u2')])
    assert len(data) == 84 + count * tri.itemsize, f'{path} is not a binary STL'
    corners = np.frombuffer(data, dtype=tri, count=count, offset=84)['corners'].reshape(-1, 3)
    vertices, index = np.unique(corners, axis=0, return_inverse=True)
    return vertices.astype(np.float32), index.reshape(-1, 3).astype(np.uint32)


def save_stl(path, vertices, triangles):
    # (V, 3) vertices, (T, 3) triangle vertex indices -> binary STL (normals left zero: viewers compute their own)
    corners = np.asarray(vertices, dtype='<f4')[np.asarray(triangles)]
    tri = np.zeros(len(triangles), dtype=np.dtype([('normal', '<f4', 3), ('corners', '<f4', (3, 3)), ('attr', '<u2')]))
    tri['corners'] = corners
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b'\0' * 80 + struct.pack('<I', len(tri)) + tri.tobytes())


class Recorder:
    # frames are kept in memory and written by save(): a minute of a two-robot scene is a few MB
    def __init__(self, name, scene, seconds_per_frame):
        self.path = LOGS_DIR / f'{name}.simrec'
        self.scene = scene
        self.seconds_per_frame = seconds_per_frame
        self.num_bodies = len(scene['bodies'])
        self.frames = []  # [(1 + 7 N,) float32 array]

    def add(self, time, poses):
        # poses: (N, 7) array-like for one env
        poses = np.asarray(poses, dtype=np.float32)
        assert poses.shape == (self.num_bodies, 7)
        self.frames.append(np.concatenate([np.float32([time]), poses.reshape(-1)]))

    def save(self, info=None):
        # write logs/{name}.simrec with every mesh the scene's bodies show
        names = sorted({v['mesh'] for b in self.scene['bodies'] for v in b.get('visuals', [])})  # [str]
        meshes = [load_stl(ASSETS_DIR / n) for n in names]  # [(vertices, triangles)]
        header = json.dumps({'scene': self.scene, 'meshes': [{'name': n, 'vertices': len(v), 'triangles': len(t)} for n, (v, t) in zip(names, meshes)],
                             'seconds_per_frame': self.seconds_per_frame, 'frames': len(self.frames), 'info': info or {}}).encode()
        header += b' ' * (-len(header) % 4)
        LOGS_DIR.mkdir(exist_ok=True)
        with gzip.open(self.path, 'wb', compresslevel=6) as f:
            f.write(MAGIC + struct.pack('<I', len(header)) + header)
            for v, t in meshes:
                f.write(v.astype('<f4').tobytes() + t.astype('<u4').tobytes())
            f.write(np.stack(self.frames).astype('<f4').tobytes())
        print(f'wrote {self.path} ({len(self.frames)} frames, {len(meshes)} meshes, {self.path.stat().st_size / 1e6:.1f} MB)')


def load(path):
    # read a .simrec -> scene, info, seconds_per_frame, (F, 1 + 7 N) float32 frames (meshes are skipped: viewers
    # in python render the collision shapes)
    with gzip.open(path, 'rb') as f:
        data = f.read()
    assert data[:8] == MAGIC, f'{path} is not a recording'
    header_len = struct.unpack_from('<I', data, 8)[0]
    header = json.loads(data[12:12 + header_len])
    offset = 12 + header_len + sum(m['vertices'] * 12 + m['triangles'] * 12 for m in header['meshes'])
    stride = 1 + 7 * len(header['scene']['bodies'])
    frames = np.frombuffer(data, dtype='<f4', offset=offset).reshape(header['frames'], stride)
    return header['scene'], header['info'], header['seconds_per_frame'], frames
