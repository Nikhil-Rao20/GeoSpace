import json
import math
import random
from pathlib import Path

import numpy as np
import rasterio
import torch

from src.augmentations import get_train_augmentations, get_val_augmentations
from src.dataloader import build_dataloaders
from src.losses import LAMBDA_WEIGHTS, build_criteria, compute_total_loss
from src.model import MultiHeadSegFormer, count_parameters
from src.utils import NUM_CLASSES, set_seed


HEAD_NAMES = ["Building", "Road", "WB_Polygon", "WB_Line", "Utility_Polygon", "Bridge"]
HEAD_TO_CHANNEL = {0: 0, 1: 1, 2: 3, 3: 4, 4: 7, 5: 8}


def has_nan_or_inf_tensor(x: torch.Tensor) -> bool:
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()


def group_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        n = p.grad.detach().data.norm(2).item()
        total += n * n
    return math.sqrt(total)


def compute_sample_weights(train_split_file, masks_dir, sample_size=500):
    names = [x.strip() for x in Path(train_split_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    rng = random.Random(42)
    sample = names if len(names) <= sample_size else rng.sample(names, sample_size)

    counts = {str(i): [0 for _ in range(NUM_CLASSES[i])] for i in range(6)}
    totals = {str(i): 0 for i in range(6)}

    for name in sample:
        with rasterio.open(Path(masks_dir) / name) as src:
            mask = src.read()
        for head_idx, ch in HEAD_TO_CHANNEL.items():
            arr = mask[ch]
            totals[str(head_idx)] += arr.size
            uniq, cnt = np.unique(arr, return_counts=True)
            for u, c in zip(uniq.tolist(), cnt.tolist()):
                if 0 <= u < NUM_CLASSES[head_idx]:
                    counts[str(head_idx)][u] += c

    weights = {}
    for h in range(6):
        total_pixels = totals[str(h)]
        n_classes = NUM_CLASSES[h]
        w = []
        for c in range(n_classes):
            cc = counts[str(h)][c]
            if cc <= 0:
                wk = 10.0
            else:
                wk = total_pixels / (n_classes * cc)
            wk = float(np.clip(wk, 0.1, 10.0))
            w.append(wk)
        w[0] = float(np.clip(0.5 * w[0], 0.1, 10.0))
        weights[str(h)] = w

    return {
        "head_names": HEAD_NAMES,
        "num_classes": NUM_CLASSES,
        "head_to_channel": HEAD_TO_CHANNEL,
        "weights": weights,
        "sample_size": len(sample),
        "mode": "sample_500",
    }


def print_weight_table(weights_payload):
    print("Class weight table:")
    for h_idx, h_name in enumerate(HEAD_NAMES):
        vals = weights_payload["weights"][str(h_idx)]
        vals_str = ", ".join([f"c{c}={v:.4f}" for c, v in enumerate(vals)])
        print(f"  {h_name}: {vals_str}")


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pipeline_pass = False
    model_forward_pass = False
    loss_pass = False
    optimizer_pass = False
    any_nan_inf = False

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        train_split_file="preprocessed_dataset/train.txt",
        val_split_file="preprocessed_dataset/val.txt",
        images_dir="preprocessed_dataset/images",
        masks_dir="preprocessed_dataset/masks",
        train_transform=get_train_augmentations(),
        val_transform=get_val_augmentations(),
        batch_size=16,
        num_workers=4,
    )

    # 1) Pipeline verification only
    batch = None
    for b in train_loader:
        if b is not None:
            batch = b
            break

    if batch is None:
        print("Pipeline batch load: FAIL (no valid batch)")
    else:
        images, masks = batch
        print("\n[1] Pipeline verification")
        print(f"Image shape: {tuple(images.shape)}")
        print(f"Image dtype: {images.dtype}")
        print(f"Image value range: [{images.min().item():.4f}, {images.max().item():.4f}]")

        assert_results = []
        assert_results.append(("Image shape == (16,3,512,512)", tuple(images.shape) == (16, 3, 512, 512)))
        assert_results.append(("Image dtype == float32", images.dtype == torch.float32))
        assert_results.append(("Image range in ~[-2.5,2.5]", images.min().item() >= -5.0 and images.max().item() <= 5.0))

        for i, m in enumerate(masks):
            print(f"Mask[{i}] shape: {tuple(m.shape)} dtype: {m.dtype} range: [{m.min().item()}, {m.max().item()}]")
            assert_results.append((f"Mask[{i}] shape == (16,512,512)", tuple(m.shape) == (16, 512, 512)))
            assert_results.append((f"Mask[{i}] dtype == int64", m.dtype == torch.int64))
            assert_results.append((f"Mask[{i}] class range valid", int(m.min().item()) >= 0 and int(m.max().item()) <= NUM_CLASSES[i] - 1))

        for msg, ok in assert_results:
            print(f"{msg}: {'PASS' if ok else 'FAIL'}")

        pipeline_pass = all(ok for _, ok in assert_results)

    # 5) Class weights check (cache or sample 500)
    class_weights_path = Path("logs_eswar/class_weights.json")
    if class_weights_path.exists():
        class_weights = json.loads(class_weights_path.read_text(encoding="utf-8"))
        class_weights["mode"] = "cached"
        print("\n[5] Loaded existing logs_eswar/class_weights.json")
    else:
        class_weights = compute_sample_weights(
            train_split_file="preprocessed_dataset/train.txt",
            masks_dir="preprocessed_dataset/masks",
            sample_size=500,
        )
        print("\n[5] Computed class weights on sample of 500 training patches")
    print_weight_table(class_weights)

    # 2) Model load + random forward pass
    print("\n[2] Model load and random forward")
    model = MultiHeadSegFormer(pretrained_model_name="nvidia/mit-b3").to(device)
    total_params = count_parameters(model)
    encoder_params = count_parameters(model.encoder)
    print(f"Total parameter count: {total_params:,}")
    print(f"Encoder parameter count: {encoder_params:,}")
    for i, h in enumerate(HEAD_NAMES):
        print(f"{h} head parameters: {count_parameters(model.heads[i]):,}")

    try:
        x = torch.randn(2, 3, 512, 512, device=device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(x)
        for i, y in enumerate(out):
            print(f"Output head[{i}] ({HEAD_NAMES[i]}): {tuple(y.shape)}")
        model_forward_pass = True
    except Exception as exc:
        print(f"Model forward FAIL: {exc}")

    used_gb = 0.0
    total_gb = 24.0
    if device.type == "cuda":
        used_gb = torch.cuda.memory_allocated() / (1024**3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU memory used after random forward: {used_gb:.2f} GB")

    # 3) Loss for one batch only + 4) one optimizer step
    print("\n[3+4] One-batch loss and one optimizer step")
    if batch is None:
        print("Cannot run loss/optimizer step: no batch")
    else:
        images, masks = batch
        images = images.to(device, non_blocking=True)
        masks = [m.to(device, non_blocking=True) for m in masks]

        criteria = build_criteria(class_weights, device)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": 6e-5, "weight_decay": 0.01, "name": "encoder"},
                {
                    "params": list(model.neck.parameters()) + list(model.heads.parameters()),
                    "lr": 3e-4,
                    "weight_decay": 0.01,
                    "name": "heads",
                },
            ]
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        model.train()
        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                total_loss, per_head_losses = compute_total_loss(logits, masks, criteria, LAMBDA_WEIGHTS)

            print(f"Total loss: {total_loss.item():.6f}")
            for k, v in per_head_losses.items():
                print(f"{k} loss: {v:.6f}")

            scalar_vals = [float(total_loss.item())] + [float(v) for v in per_head_losses.values()]
            any_nan_inf = any((not math.isfinite(v)) for v in scalar_vals)
            print(f"NaN/Inf present: {'Yes' if any_nan_inf else 'No'}")
            loss_pass = not any_nan_inf
        except Exception as exc:
            print(f"Loss compute FAIL: {exc}")
            loss_pass = False

        if loss_pass:
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                optimizer.zero_grad(set_to_none=True)
                micro = 1
                bsz = images.shape[0]
                n_chunks = math.ceil(bsz / micro)

                for s in range(0, bsz, micro):
                    e = min(s + micro, bsz)
                    img_mb = images[s:e]
                    msk_mb = [m[s:e] for m in masks]

                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        logits_mb = model(img_mb)
                        loss_mb, _ = compute_total_loss(logits_mb, msk_mb, criteria, LAMBDA_WEIGHTS)
                        loss_mb = loss_mb / n_chunks

                    scaler.scale(loss_mb).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                enc_norm = group_grad_norm(list(model.encoder.parameters()))
                head_norm = group_grad_norm(list(model.neck.parameters()) + list(model.heads.parameters()))

                scaler.step(optimizer)
                scaler.update()

                print("One optimizer step completed successfully")
                print(f"Gradient norm (encoder group): {enc_norm:.6f}")
                print(f"Gradient norm (heads group): {head_norm:.6f}")
                optimizer_pass = True
            except Exception as exc:
                print(f"Optimizer step FAIL: {exc}")

    ready_to_train = pipeline_pass and model_forward_pass and loss_pass and optimizer_pass and (not any_nan_inf)

    print("\nVERIFICATION COMPLETE")
    print("=====================")
    print(f"Pipeline:        {'PASS' if pipeline_pass else 'FAIL'}")
    print(f"Model forward:   {'PASS' if model_forward_pass else 'FAIL'}")
    print(f"Loss:            {'PASS' if loss_pass else 'FAIL'}")
    print(f"Optimizer step:  {'PASS' if optimizer_pass else 'FAIL'}")
    print(f"Any NaN/Inf:     {'Yes' if any_nan_inf else 'No'}")
    print(f"GPU memory used: {used_gb:.2f} GB / {total_gb:.2f} GB")
    print(f"Ready to train:  {'Yes' if ready_to_train else 'No'}")


if __name__ == "__main__":
    main()