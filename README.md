<div align="center">
  <h2>driving-idm</h2>
</div>

Takes frame `t` and frame `t+1`, predicts the action between them.

### Setup & Run

Requires two Modal secrets:
- `huggingface-secret` — `HUGGING_FACE_HUB_TOKEN`
- `wandb-secret` — `WANDB_API_KEY`

```bash
uv run --with modal modal run --detach train.py [--run-name my-run]

modal app logs idm-training
```

### Dataset

`nebusoku14/comm_hack_parking_npz` — ~300–400 `.npz` files, each with:
- `frames`: `(120, 180, 320, 3)` uint8
- `actions`: `(120, 2)` float32 (steering, acceleration)

### Model

4× strided ConvBlocks → AdaptiveAvgPool → MLP head. ~500K params. Input is 6-channel stacked frame pair resized to `90×160`.
