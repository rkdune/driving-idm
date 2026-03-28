# IDM Training

Trains a lightweight Inverse Dynamics Model (IDM) on parking footage. Takes frame `t` and frame `t+1`, predicts the action between them.

## Run

```bash
uv run --with modal modal run --detach train.py [--run-name my-run]
```

The `--detach` flag prevents `ConflictError: Function call has expired` on long runs — the local process exits immediately after spawning and training continues on Modal. Monitor with:

```bash
modal app logs idm-training
```

## Setup

Requires two Modal secrets:
- `huggingface-secret` — `HUGGING_FACE_HUB_TOKEN`
- `wandb-secret` — `WANDB_API_KEY`

## Dataset

`nebusoku14/comm_hack_parking_npz` — ~300–400 `.npz` files, each with:
- `frames`: `(120, 180, 320, 3)` uint8
- `actions`: `(120, 2)` float32 (steering, acceleration)

## Model

4× strided ConvBlocks → AdaptiveAvgPool → MLP head. ~500K params. Input is 6-channel stacked frame pair resized to `90×160`.
