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

### Rollout (video model + IDM)

Chains a video model with the IDM: video model predicts frame `t+1` from past context, IDM predicts the action from `(frame_t, predicted_frame_t+1)`.

```bash
uv run --with modal modal run --detach rollout.py \
    --idm-ckpt /path/to/idm_best.pt \
    --video-model-ckpt /path/to/video_model.pt \
    [--run-name my-rollout] \
    [--context-frames 8] \
    [--n-eval-sequences 200]
```

Omit `--video-model-ckpt` to run with the `LastFrameBaseline` (next frame = current frame).

Three metrics logged to wandb:
- `vm_idm/*` — main pipeline (video model → IDM) vs ground truth
- `oracle_idm/*` — IDM with real next frame (upper bound)
- `baseline/*` — IDM with no motion (lower bound)

To plug in a custom architecture, subclass `CheckpointVideoModel` in `rollout.py` and implement `build_model()`.
