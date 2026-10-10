# Backend

Batched rigid-body physics engine in PyTorch (quaternions, joints with motors and limits, contacts with friction and
omni wheels, camera rendering), the LeKiwi robot (omni base + SO-101 arm), its pick-and-place environment and the
learning pipeline.
See `../documentation.md` for how everything works.

Run everything from `back/` as modules:

```bash
pip install -r requirements.txt
python -m write_scenes               # scenes/lekiwi_pick.json
python -m api                        # server on http://localhost:8080 (or `python main.py` from the repo root)

python -m ml.collect --envs 128 --episodes 500        # expert demonstrations -> data/bc/
python -m ml.train_bc --steps 30000                    # -> checkpoints/bc.pt
python -m ml.train_ppo --init checkpoints/bc.pt        # -> checkpoints/ppo.pt
python -m ml.eval --ckpt checkpoints/ppo.pt            # success rate + logs/policy_rollout/
```

Requirements: `requirements.txt`
