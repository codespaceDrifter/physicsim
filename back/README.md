# Backend

Batched rigid-body physics engine in PyTorch (quaternions, joints with motors and limits, contacts with friction,
camera rendering), the ToFuBot robot, its pick-and-place environment and the learning pipeline.
See `../documentation.md` for how everything works.

Run everything from `back/` as modules:

```bash
pip install fastapi uvicorn
python -m simulate                   # write scenes/*.json, record three_body + billiards into logs/
python -m api                        # server on http://localhost:8080 (frontend: cd ../front && npm run dev)

python -m ml.collect --envs 128 --episodes 500        # expert demonstrations -> data/bc/
python -m ml.train_bc --steps 30000                    # -> checkpoints/bc.pt
python -m ml.train_ppo --init checkpoints/bc.pt        # -> checkpoints/ppo.pt
python -m ml.eval --ckpt checkpoints/ppo.pt            # success rate + logs/policy_rollout/
```

Requirements: `torch numpy fastapi uvicorn`
