import modal

app = modal.App("idm-training")

volume = modal.Volume.from_name("idm-dataset-cache", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "wandb",
        "huggingface_hub",
        "tqdm",
    )
)


@app.function(
    image=image,
    gpu="H100",
    cpu=64,
    timeout=3600 * 4,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    memory=131072,
    volumes={VOLUME_PATH: volume},
)
def train(run_name: str | None = None, clear_cache: bool = False, curvature_comma: bool = False, diff_siamese: bool = False):
    import os
    import time
    import math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from huggingface_hub import hf_hub_download, list_repo_files
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import wandb
    from tqdm import tqdm

    # ── Config ───────────────────────────────────────────────────────────────
    REPO_ID = "nebusoku14/comm_hack_parking_day" if curvature_comma else "nebusoku14/comm_hack_parking_npz"
    IMG_H, IMG_W = 90, 160
    BATCH_SIZE = 128           # bigger batch for H100
    LR = 3e-4
    EPOCHS = 1
    NUM_WORKERS = 4
    LOG_EVERY     = 50         # steps between per-step train wandb logs
    VAL_LOG_EVERY = 50         # steps between mid-epoch val probes
    VAL_PROBE_BATCHES = 20     # how many val batches to sample for the probe
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    run = wandb.init(
        project="idm-parking",
        name=run_name,
        config=dict(
            img_h=IMG_H, img_w=IMG_W,
            batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS,
            log_every=LOG_EVERY,
            model="diff-siamese-idm" if diff_siamese else "resnet-idm",
            dataset=REPO_ID,
            gpu="H100",
        ),
    )
    print(f"wandb run: {run.url}")
    print(
        f"\n{'='*60}\n"
        f"  DATASET:      {REPO_ID}\n"
        f"  MODEL:        {'DIFF-SIAMESE-IDM' if diff_siamese else 'RESNET-IDM'}\n"
        f"  BATCH SIZE:   {BATCH_SIZE}\n"
        f"  LR:           {LR}\n"
        f"  EPOCHS:       {EPOCHS}\n"
        f"  IMG:          {IMG_H}x{IMG_W}\n"
        f"  CURVATURE:    {curvature_comma}\n"
        f"  DIFF_SIAMESE: {diff_siamese}\n"
        f"{'='*60}\n"
    )

    # ── Download dataset ─────────────────────────────────────────────────────
    print("Listing dataset files...")
    all_files = sorted(
        f for f in list_repo_files(REPO_ID, repo_type="dataset")
        if f.endswith(".npz")
    )
    print(f"Found {len(all_files)} npz files")
    wandb.config.update({"num_npz_files": len(all_files)})

    npz_dir = os.path.join(VOLUME_PATH, "npz")
    os.makedirs(npz_dir, exist_ok=True)

    # Optionally wipe the cache and re-download everything
    if clear_cache:
        print("Clearing volume cache...")
        for fname in os.listdir(npz_dir):
            fpath = os.path.join(npz_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        volume.commit()
        print("Cache cleared.")

    # Only download files not already on the volume
    existing = set(os.listdir(npz_dir))
    to_download = [f for f in all_files if os.path.basename(f) not in existing]
    print(f"Already cached: {len(existing)}  |  To download: {len(to_download)}")

    t0 = time.time()
    if to_download:
        tmp_dir = "/tmp/npz_staging"
        os.makedirs(tmp_dir, exist_ok=True)

        def _download(fname):
            hf_hub_download(repo_id=REPO_ID, filename=fname,
                            repo_type="dataset", local_dir=tmp_dir)
            return fname

        with ThreadPoolExecutor(max_workers=64) as ex:
            futures = {ex.submit(_download, f): f for f in to_download}
            for fut in tqdm(as_completed(futures), total=len(to_download), desc="download"):
                fut.result()

        print("Moving files to volume...")
        import shutil
        for fname in to_download:
            src = os.path.join(tmp_dir, os.path.basename(fname))
            dst = os.path.join(npz_dir, os.path.basename(fname))
            shutil.move(src, dst)
        volume.commit()
        print(f"Download done in {time.time() - t0:.1f}s")
    else:
        print("All files already cached, skipping download.")

    local_paths = [os.path.join(npz_dir, os.path.basename(f)) for f in all_files]

    # ── Preload into RAM ─────────────────────────────────────────────────────
    print("Loading npz files into RAM...")

    def _load(p):
        try:
            data = np.load(p)
            return data["frames"], data["actions"]
        except Exception as e:
            return None, str(e)

    with ThreadPoolExecutor(max_workers=64) as ex:
        results = list(tqdm(ex.map(_load, local_paths), total=len(local_paths), desc="load"))

    all_frames, all_actions, skipped = [], [], 0
    for p, (frames, actions) in zip(local_paths, results):
        if frames is None:
            print(f"  skip {os.path.basename(p)}: {actions}")
            skipped += 1
        else:
            all_frames.append(frames)
            all_actions.append(actions)

    frame_counts = [f.shape[0] for f in all_frames]
    print(f"Loaded {len(all_frames)} files ({skipped} skipped)")
    print(f"Frames per file — min: {min(frame_counts)}  max: {max(frame_counts)}  "
          f"mean: {sum(frame_counts)/len(frame_counts):.1f}  total: {sum(frame_counts):,}")

    # Compute action stats for normalisation + logging
    all_act_flat = np.concatenate(all_actions, axis=0)   # (N, 2)
    act_mean = all_act_flat.mean(axis=0)
    act_std  = all_act_flat.std(axis=0) + 1e-6
    print(f"Action mean: {act_mean}  std: {act_std}")
    wandb.config.update({
        "action_mean": act_mean.tolist(),
        "action_std":  act_std.tolist(),
        "action_min":  all_act_flat.min(axis=0).tolist(),
        "action_max":  all_act_flat.max(axis=0).tolist(),
    })

    # Build flat (file_idx, t) sample index
    sample_index = [
        (i, t)
        for i, f in enumerate(all_frames)
        for t in range(f.shape[0] - 1)
    ]
    print(f"Total samples: {len(sample_index):,}")
    wandb.config.update({"total_samples": len(sample_index)})

    # ── Dataset ──────────────────────────────────────────────────────────────
    class IDMDataset(Dataset):
        def __init__(self, indices, frames_list, actions_list,
                     img_h, img_w, act_mean, act_std):
            self.indices = indices
            self.frames  = frames_list
            self.actions = actions_list
            self.img_h, self.img_w = img_h, img_w
            self.act_mean = torch.from_numpy(act_mean)
            self.act_std  = torch.from_numpy(act_std)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            fi, t = self.indices[idx]
            f0  = self.frames[fi][t]
            f1  = self.frames[fi][t + 1]
            act = self.actions[fi][t]

            f0 = torch.from_numpy(f0).permute(2, 0, 1).float() / 255.0
            f1 = torch.from_numpy(f1).permute(2, 0, 1).float() / 255.0

            f0 = F.interpolate(f0.unsqueeze(0), size=(self.img_h, self.img_w),
                               mode="bilinear", align_corners=False).squeeze(0)
            f1 = F.interpolate(f1.unsqueeze(0), size=(self.img_h, self.img_w),
                               mode="bilinear", align_corners=False).squeeze(0)

            x = torch.cat([f0, f1], dim=0)   # (6, H, W)
            y = (torch.from_numpy(act.copy()) - self.act_mean) / self.act_std
            return x, y

    # 90/10 split by file (shuffled so val isn't just the last N files)
    rng = np.random.default_rng()
    file_indices = np.arange(len(all_frames))
    rng.shuffle(file_indices)
    split = int(0.9 * len(file_indices))
    train_files = set(file_indices[:split].tolist())
    val_files   = set(file_indices[split:].tolist())
    train_idx = [(fi, t) for fi, t in sample_index if fi in train_files]
    val_idx   = [(fi, t) for fi, t in sample_index if fi in val_files]

    train_ds = IDMDataset(train_idx, all_frames, all_actions,
                          IMG_H, IMG_W, act_mean, act_std)
    val_ds   = IDMDataset(val_idx,   all_frames, all_actions,
                          IMG_H, IMG_W, act_mean, act_std)
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")
    wandb.config.update({"train_samples": len(train_ds), "val_samples": len(val_ds)})

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)
            self.skip  = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

        def forward(self, x):
            return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x))

    class IDM(nn.Module):
        """Stacked-frame IDM: (frame_t ‖ frame_t+1) → action_t (normalised)"""
        def __init__(self, action_dim=2):
            super().__init__()
            self.encoder = nn.Sequential(
                ResBlock(6,   32,  stride=2),   # 45×80
                ResBlock(32,  64,  stride=2),   # 23×40
                ResBlock(64,  128, stride=2),   # 12×20
                ResBlock(128, 256, stride=2),   #  6×10
                ResBlock(256, 512, stride=1),   #  6×10  deeper before pooling
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, action_dim),
            )

        def forward(self, x):
            return self.head(self.encoder(x).flatten(1))

    class IDMSiamese(nn.Module):
        """Siamese IDM: f0 and f1 encoded with shared weights.
        Head sees [z0, z1, z1-z0] — explicit feature-level diff signal."""
        def __init__(self, action_dim=2):
            super().__init__()
            self.encoder = nn.Sequential(
                ResBlock(3,   32,  stride=2),   # 45×80
                ResBlock(32,  64,  stride=2),   # 23×40
                ResBlock(64,  128, stride=2),   # 12×20
                ResBlock(128, 256, stride=2),   #  6×10
                ResBlock(256, 512, stride=1),   #  6×10  deeper before pooling
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Sequential(
                nn.Linear(512 * 3, 256),        # z0, z1, z1-z0
                nn.ReLU(inplace=True),
                nn.Linear(256, action_dim),
            )

        def forward(self, x):
            f0, f1 = x[:, :3], x[:, 3:]
            z0 = self.encoder(f0).flatten(1)
            z1 = self.encoder(f1).flatten(1)
            return self.head(torch.cat([z0, z1, z1 - z0], dim=1))

    model = (IDMSiamese(action_dim=2) if diff_siamese else IDM(action_dim=2)).to(DEVICE)
    # Use torch.compile on H100 for speed
    model = torch.compile(model)
    wandb.watch(model, log="all", log_freq=LOG_EVERY)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    wandb.config.update({"model_params": n_params})

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        total_steps=total_steps, pct_start=0.05,
    )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def compute_metrics(pred, target, act_std_t):
        """Returns dict of per-dim and aggregate metrics (unnormalised MAE, MSE)."""
        mse_norm  = F.mse_loss(pred, target).item()
        mae_norm  = (pred - target).abs().mean().item()
        # unnormalise
        pred_un   = pred   * act_std_t + act_mean_t
        target_un = target * act_std_t + act_mean_t
        mse_un    = F.mse_loss(pred_un, target_un).item()
        mae_un    = (pred_un - target_un).abs().mean().item()
        mae_dim   = (pred_un - target_un).abs().mean(dim=0)   # (2,)
        return dict(
            mse_norm=mse_norm, mae_norm=mae_norm,
            mse=mse_un, mae=mae_un,
            mae_steer=mae_dim[0].item(), mae_accel=mae_dim[1].item(),
        )

    act_std_t = torch.from_numpy(act_std).to(DEVICE)
    act_mean_t = torch.from_numpy(act_mean).to(DEVICE)

    def quick_val(n_batches):
        """Run over up to n_batches of val data and return averaged metrics."""
        model.eval()
        totals = dict(mse_norm=0., mae=0., mae_steer=0., mae_accel=0.)
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                pred = model(x)
                m = compute_metrics(pred, y, act_std_t)
                for k in totals:
                    totals[k] += m[k]
                count += 1
                if count >= n_batches:
                    break
        model.train()
        return {k: v / count for k, v in totals.items()}

    # ── Train / Val loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    global_step   = 0
    train_start   = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ---- train ----
        model.train()
        running_loss = 0.0
        running_mae  = 0.0
        batch_times  = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", dynamic_ncols=True)
        for step, (x, y) in enumerate(pbar):
            t_batch = time.time()
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            pred = model(x)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            mae_val  = (pred.detach() - y).abs().mean().item()
            running_loss += loss_val
            running_mae  += mae_val
            batch_times.append(time.time() - t_batch)
            global_step  += 1

            pbar.set_postfix(loss=f"{loss_val:.4f}", mae=f"{mae_val:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if global_step % LOG_EVERY == 0:
                samples_per_sec = BATCH_SIZE / (sum(batch_times) / len(batch_times))
                wandb.log({
                    "step": global_step,
                    "train/loss_step":       loss_val,
                    "train/mae_step":        mae_val,
                    "train/grad_norm":       grad_norm.item(),
                    "train/samples_per_sec": samples_per_sec,
                    "lr": scheduler.get_last_lr()[0],
                }, step=global_step)
                batch_times = []

            if global_step % VAL_LOG_EVERY == 0:
                vm = quick_val(VAL_PROBE_BATCHES)
                print(f"  [step {global_step}] val probe — loss: {vm['mse_norm']:.4f}  "
                      f"mae: {vm['mae']:.4f}  steer: {vm['mae_steer']:.4f}  accel: {vm['mae_accel']:.4f}")
                wandb.log({
                    "step": global_step,
                    "val/loss_step":       vm["mse_norm"],
                    "val/mae_step":        vm["mae"],
                    "val/mae_steer_step":  vm["mae_steer"],
                    "val/mae_accel_step":  vm["mae_accel"],
                }, step=global_step)

        n_batches  = len(train_loader)
        train_loss = running_loss / n_batches
        train_mae  = running_mae  / n_batches
        epoch_time = time.time() - epoch_start

        # ---- val ----
        model.eval()
        val_loss = val_mae = 0.0
        val_mae_steer = val_mae_accel = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]", dynamic_ncols=True):
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                pred = model(x)
                m = compute_metrics(pred, y, act_std_t)
                val_loss      += m["mse_norm"]
                val_mae       += m["mae"]
                val_mae_steer += m["mae_steer"]
                val_mae_accel += m["mae_accel"]
                val_batches   += 1

        val_loss      /= val_batches
        val_mae       /= val_batches
        val_mae_steer /= val_batches
        val_mae_accel /= val_batches

        elapsed_total = time.time() - train_start
        print(
            f"\n{'='*70}\n"
            f"Epoch {epoch}/{EPOCHS}  ({epoch_time:.1f}s, total {elapsed_total/60:.1f}min)\n"
            f"  train loss (MSE norm): {train_loss:.5f}  |  train MAE norm: {train_mae:.5f}\n"
            f"  val   loss (MSE norm): {val_loss:.5f}  |  val MAE unnorm: {val_mae:.4f}\n"
            f"    steer MAE: {val_mae_steer:.4f}  |  accel MAE: {val_mae_accel:.4f}\n"
            f"  lr: {scheduler.get_last_lr()[0]:.2e}  |  best val: {best_val_loss:.5f}\n"
            f"{'='*70}"
        )

        wandb.log({
            "epoch": epoch,
            "train/loss":     train_loss,
            "train/mae_norm": train_mae,
            "val/loss":       val_loss,
            "val/mae":        val_mae,
            "val/mae_steer":  val_mae_steer,
            "val/mae_accel":  val_mae_accel,
            "epoch_time_s":   epoch_time,
            "total_time_min": elapsed_total / 60,
            "lr": scheduler.get_last_lr()[0],
        }, step=global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = "/tmp/idm_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "act_mean": act_mean,
                "act_std":  act_std,
            }, ckpt_path)
            wandb.save(ckpt_path)
            print(f"  ✓ saved best checkpoint (val_loss={val_loss:.5f})")

    wandb.finish()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")


@app.local_entrypoint()
def main(run_name: str = "", clear_cache: bool = False, curvature_comma: bool = False, diff_siamese: bool = False):
    # Use spawn so the local process doesn't hold an open connection.
    # Run with: modal run --detach train.py [--run-name my-run] [--clear-cache] [--curvature-comma] [--diff-siamese]
    call = train.spawn(run_name=run_name or None, clear_cache=clear_cache, curvature_comma=curvature_comma, diff_siamese=diff_siamese)
    print(f"Spawned function call: {call.object_id}")
    print("Track progress in the Modal dashboard or with: modal app logs idm-training")
