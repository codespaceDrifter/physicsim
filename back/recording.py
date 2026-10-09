"""
Recording system - matches Go binary format exactly.
"""
import struct
from pathlib import Path
from world import World


class Recording:
    def __init__(self, name: str, world: World, seconds_per_frame: float, end_seconds: int):
        """
        Args:
            name: Recording name
            world: World to simulate
            seconds_per_frame: Time between recorded frames
            end_seconds: Total simulation duration
        """
        self.name = name
        self.world = world
        self.seconds_per_frame = seconds_per_frame
        self.end_seconds = end_seconds

        # Create recording directory (one level up from python-back)
        self.recording_dir = Path(__file__).parent.parent / 'logs' / name
        self.recording_dir.mkdir(parents=True, exist_ok=True)

    def simulate(self):
        """Run simulation and save recordings."""
        print("Simulating recording...")

        total_frames = int(self.end_seconds / self.seconds_per_frame) + 1
        print(f"Total frames: {total_frames}")

        frames = {}  # Dict of time -> frame_data
        chunk_index = 0

        for i in range(total_frames):
            if i % 100 == 0:
                print(f"Frame: {i}")

            # Run simulation until target time
            target_time = i * self.seconds_per_frame
            while self.world.time < target_time:
                self.world.update()

            # Record frame
            frame_data = self.world.get_frame_data()
            time_key = f"{target_time:.3f}"
            frames[time_key] = frame_data

            # Save chunk every 1000 frames or at end
            if len(frames) == 1000 or i == total_frames - 1:
                self._save_chunk(chunk_index, frames, total_frames)
                frames = {}
                chunk_index += 1

        print(f"Recording complete! Saved {chunk_index} chunks")

    def _save_chunk(self, chunk_index: int, frames: dict, total_frames: int):
        """Save chunk in Go-compatible binary format."""
        chunk_path = self.recording_dir / f"{chunk_index}.bin"

        with open(chunk_path, 'wb') as f:
            # Magic header
            f.write(b'SIMB')

            # Seconds per frame
            f.write(struct.pack('<f', self.seconds_per_frame))

            # Total frames in entire recording
            f.write(struct.pack('<I', total_frames))

            # Number of frames in this chunk
            f.write(struct.pack('<I', len(frames)))

            # Sort frames by time
            sorted_times = sorted([float(k) for k in frames.keys()])

            for time in sorted_times:
                time_key = f"{time:.3f}"
                frame = frames[time_key]

                # Write frame time
                f.write(struct.pack('<f', time))

                # Number of objects
                num_objects = len(frame['id_array'])
                f.write(struct.pack('<I', num_objects))

                # Write each object
                for i in range(num_objects):
                    # ID (length-prefixed string)
                    obj_id = frame['id_array'][i].encode('utf-8')
                    f.write(struct.pack('<I', len(obj_id)))
                    f.write(obj_id)

                    # Position (3 floats)
                    pos = frame['position_arrays'][i]
                    for j in range(3):
                        f.write(struct.pack('<f', pos[j]))

                    # Vertices (count + floats)
                    verts = frame['vertex_arrays'][i]
                    f.write(struct.pack('<I', len(verts)))
                    for v in verts:
                        f.write(struct.pack('<f', v))

                    # Indices (count + uints)
                    inds = frame['index_arrays'][i]
                    f.write(struct.pack('<I', len(inds)))
                    for idx in inds:
                        f.write(struct.pack('<I', idx))

                    # Color (length-prefixed string)
                    color = frame['color_array'][i].encode('utf-8')
                    f.write(struct.pack('<I', len(color)))
                    f.write(color)

        print(f"Saved chunk {chunk_index} with {len(frames)} frames")