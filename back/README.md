# PyTorch Physics Backend

Tensor-based physics engine (linear dynamics only). See `../documentation.md` for the full walkthrough.

Run from `back/`:

```bash
python -m simulate   # wipe logs/ and regenerate all example recordings
python -m api        # FastAPI server on http://localhost:8080
```

Endpoints:
- `GET /recordings/{name}/{chunk}` - binary recording chunk
- `GET /worlds` - list of recordings
- `GET /health` - health check

Requirements: `torch numpy fastapi uvicorn`
