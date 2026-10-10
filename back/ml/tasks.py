# The two tasks the ML scripts can run: "pick" (one robot, pick-and-place) and "race" (two robots, block race).
from robot.pick_place import PickPlaceEnv
from robot.block_race import BlockRaceEnv

TASKS = ['pick', 'race']


def make_env(task, num_envs, device, seed=0):
    assert task in TASKS, f'unknown task {task}'
    return PickPlaceEnv(num_envs, device, num_blocks=1, seed=seed) if task == 'pick' else BlockRaceEnv(num_envs, device, seed=seed)


def good_episode(task, info, e, r):
    # an expert episode worth imitating: pick = the block was delivered, race = this robot scored at least once
    return bool(info['success'][e]) if task == 'pick' else int(info['score'][e, r]) >= 1
