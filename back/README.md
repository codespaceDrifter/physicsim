# Backend

Batched rigid-body physics engine in PyTorch (quaternions, joints with motors and limits, contacts with friction and
omni wheels, camera rendering), the LeKiwi robot (omni base + SO-101 arm), its pick-and-place environment and the
learning pipeline.
See `../documentation.md` for how everything works.

Run everything from `back/` as modules:

```bash
pip install -r requirements.txt
python -m write_scenes               # scenes/lekiwi_pick.json, scenes/block_race.json
python -m api                        # server on http://localhost:8080 (or `python main.py` from the repo root)

python -m ml.collect --task race --envs 128 --episodes 200   # expert demonstrations -> data/bc/
python -m ml.train_bc --task race --steps 30000               # -> checkpoints/bc.pt
python -m ml.train_ppo --task race --init checkpoints/bc.pt   # self-play -> checkpoints/ppo.pt
python -m ml.eval --task race --ckpt checkpoints/ppo.pt       # win rate vs the expert + logs/policy_rollout/
```

Requirements: `requirements.txt`
