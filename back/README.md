# Backend

Batched rigid-body physics engine in PyTorch (quaternions, joints with motors and limits, contacts with friction and
omni wheels, camera rendering), the LeKiwi robot (omni base + SO-101 arm), its course environment (drive, pick, bring, race, speedrun) and the
learning pipeline.
See `../documentation.md` for how everything works.

Run everything from `back/` as modules:

```bash
pip install -r requirements.txt
python -m write_scenes               # scenes/course_{1_drive,2_pick,3_bring,4_race,5_speedrun}.json
python -m api                        # server on http://localhost:8080 (or `python main.py` from the repo root)

python -m ml.train_teacher            # PPO on state, all stages -> checkpoints/teacher.pt
python -m ml.distill                  # teacher -> VLA (cloning + DAgger) -> checkpoints/vla_sft.pt
python -m ml.train_grpo               # GRPO fine-tune -> checkpoints/vla_grpo.pt
python -m ml.eval --vla checkpoints/vla_sft.pt checkpoints/vla_grpo.pt   # results table
```

Requirements: `requirements.txt`
