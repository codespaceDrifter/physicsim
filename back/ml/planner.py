# System 2: a pretrained language model turns a typed request into the body policy's skill commands. Qwen3.5-4B (text
# branch; ml/qwen is MINDOLOGY's synapse implementation with its imports adapted) is prompted with the skills and
# writes the plan one command per line, then "done". decoding is grammar-constrained and greedy: at every step only
# tokens that continue a valid line are allowed (a trie over the token ids of every line), so each line is exactly a
# sentence System 1 (robot/warehouse.py) was trained on. a walk line carries how long to walk:
#   "walk forward +0.50 left +0.00 turn +0.00 for 3 seconds"
# the executor (ml/demo_humanoid.py) runs each line until its skill succeeds (walk: for its seconds).
#
# weights: data/qwen3_5_4b, the HF Qwen/Qwen3.5-4B checkpoint (config.json, tokenizer.json, chat_template.jinja,
# model.safetensors shards; locally a link to MINDOLOGY/weights/evoke/Qwen3_5_4b). needs transformers (tokenizer),
# safetensors and flash-linear-attention (imported by ml/qwen/GatedDeltaNet.py).
#
# run from back/:  python -m ml.planner "pick up the red box and put it on the blue pad"
import argparse
import json
import time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from ml.qwen.qwen3_5 import Qwen3_5ForCausalLM
from ml.qwen.config import QWEN3_5_4B_CONFIG
from robot.humanoid import COMMAND_RANGES, COMMAND_STEP
from robot.warehouse import BOXES

WEIGHTS = Path(__file__).parent.parent / 'data' / 'qwen3_5_4b'
# longest plan (lines)
MAX_LINES = 8
# walk line durations (s)
WALK_SECONDS = range(1, 11)
SYSTEM = f'''You control a humanoid robot in a warehouse. Turn the user's request into a plan: one command per line, \
then the line "done". These are the only commands:
- walk forward F left L turn T for N seconds   (F, L in m/s, T in rad/s (positive = left), written like +0.50; \
forward {COMMAND_RANGES[0][0]:+.2f} to {COMMAND_RANGES[0][1]:+.2f}, left {COMMAND_RANGES[1][0]:+.2f} to \
{COMMAND_RANGES[1][1]:+.2f}, turn {COMMAND_RANGES[2][0]:+.2f} to {COMMAND_RANGES[2][1]:+.2f}, steps of {COMMAND_STEP}; \
N from 1 to 10)
- pick up the C box
- put the C box on the P pad   (picks the box up if it isn't held yet, carries it there, sets it down)
- run a lap   (one lap of the walled track)
- run a lap with the C box
- sort the boxes by color   (every box onto the pad of its own color)
Box and pad colors C, P: {", ".join(BOXES)}. Use as few commands as the request needs. A request with several steps gets one line per step, in order. Examples:
request: back up a little, then spin right
walk forward -0.50 left +0.00 turn +0.00 for 2 seconds
walk forward +0.00 left +0.00 turn -1.00 for 3 seconds
done
request: bring the green box over to the yellow pad
put the green box on the yellow pad
done
request: jog around the track
run a lap
done'''


def commands():
    # [str] every valid plan line (without its newline)
    grid = lambda lo, hi: [f'{v * COMMAND_STEP:+.2f}' for v in range(round(lo / COMMAND_STEP), round(hi / COMMAND_STEP) + 1)]
    (f0, f1), (l0, l1), (t0, t1) = COMMAND_RANGES
    walk = [f'walk forward {f} left {l} turn {t} for {n} seconds' for f in grid(f0, f1) for l in grid(l0, l1) for t in grid(t0, t1)
            for n in WALK_SECONDS]
    boxes = [f'pick up the {c} box' for c in BOXES] + [f'run a lap with the {c} box' for c in BOXES] + \
            [f'put the {c} box on the {p} pad' for c in BOXES for p in BOXES]
    return walk + boxes + ['run a lap', 'sort the boxes by color']


def load(device, dtype):
    # Qwen3.5-4B text model with the checkpoint's weights (vision tower keys skipped), eval mode, and its tokenizer
    model = Qwen3_5ForCausalLM(QWEN3_5_4B_CONFIG)
    own = model.state_dict()
    index = json.loads((WEIGHTS / 'model.safetensors.index.json').read_text())
    state = {}  # {key: tensor}
    for shard in sorted(set(index['weight_map'].values())):
        with safe_open(str(WEIGHTS / shard), framework='pt', device='cpu') as f:
            state.update({k: f.get_tensor(k).to(dtype) for k in f.keys() if k in own})
    missing, unexpected = model.load_state_dict(state, strict=False)
    # the lm head is tied to the embedding (no checkpoint key)
    assert missing == ['lm_head.weight'] and not unexpected, (missing, unexpected)
    model.tie_weights()
    return model.to(device=device, dtype=dtype).eval(), AutoTokenizer.from_pretrained(str(WEIGHTS), local_files_only=True)


class Planner:
    def __init__(self, device, dtype=torch.bfloat16):
        self.device = device
        self.model, self.tok = load(device, dtype)
        # trie over token ids: {token: child node}; a node's None key marks the end of a line (newline, or "done")
        self.trie = {}  # {int | None: dict}
        for line in commands():
            self.insert(self.tok.encode(line + '\n', add_special_tokens=False))
        self.insert(self.tok.encode('done', add_special_tokens=False))
        self.done = tuple(self.tok.encode('done', add_special_tokens=False))

    def insert(self, ids):
        node = self.trie
        for t in ids:
            node = node.setdefault(t, {})
        node[None] = {}

    @torch.no_grad()
    def plan(self, request):
        # request (str) -> [str] plan lines (without "done")
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': request}]
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = torch.tensor([self.tok.encode(prompt, add_special_tokens=False)], device=self.device)
        mask = torch.ones_like(ids)
        logits, _, cache = self.model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
        lines, line, node = [], [], self.trie  # [str] finished lines, [int] the current line's tokens, its trie node
        while len(lines) < MAX_LINES:
            allowed = [t for t in node if t is not None]
            assert allowed
            # (vocab,) next-token logits, everything but the allowed continuations masked out
            scores = logits[0, -1].float()
            token = allowed[int(scores[torch.tensor(allowed, device=self.device)].argmax())]
            line.append(token)
            node = node[token]
            if None in node and not [t for t in node if t is not None]:
                if tuple(line) == self.done:
                    break
                lines.append(self.tok.decode(line).strip())
                line, node = [], self.trie
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
            logits, _, cache = self.model(input_ids=torch.tensor([[token]], device=self.device), attention_mask=mask,
                                          past_key_values=cache, logits_to_keep=1)
        return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('request', nargs='+')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    t0 = time.time()
    planner = Planner(torch.device(args.device))
    print(f'loaded in {time.time() - t0:.0f} s')
    for request in args.request:
        t0 = time.time()
        print(f'{request!r} -> {planner.plan(request)} ({time.time() - t0:.1f} s)')
