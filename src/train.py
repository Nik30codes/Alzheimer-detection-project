"""
Training loop. PyTorch doesn't have Keras-style callback objects (EarlyStopping,
ReduceLROnPlateau) built into model.fit() -- here they're just written out
directly as plain Python, which ends up being about the same amount of code.

Uses automatic mixed precision (autocast + GradScaler): does most of the
forward/backward pass in float16 instead of float32, which roughly halves
memory use and speeds up training -- worth it specifically because the GPU
here has only 6GB VRAM.
"""

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.amp import autocast, GradScaler


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, freeze_bn=False):
    """One pass over a DataLoader. Trains if optimizer is given, otherwise just evaluates."""
    is_training = optimizer is not None
    if is_training:
        model.train()
        if freeze_bn:
            from models import freeze_batchnorm
            freeze_batchnorm(model)  # model.train() just reset BN to train mode -- reassert eval
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_true, all_pred = [], []
    with torch.set_grad_enabled(is_training):
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_true.append(labels.detach().cpu())
            all_pred.append(preds.detach().cpu())

    y_true = torch.cat(all_true).numpy() if all_true else None
    y_pred = torch.cat(all_pred).numpy() if all_pred else None
    return total_loss / total, correct / total, y_true, y_pred


def train_model(model, train_loader, val_loader, class_weights, device,
                 epochs=30, lr=1e-3, patience=5, checkpoint_path="models/checkpoints/best.pt",
                 initial_best_val_loss=float("inf"), weight_decay=0.0, freeze_bn=False,
                 select_by="val_loss"):
    """
    Trains until either `epochs` is reached or the monitored validation metric hasn't
    improved for `patience` epochs (early stopping). Always reloads the best
    checkpoint before returning -- equivalent to Keras's restore_best_weights=True.

    select_by chooses WHICH validation metric decides "best":
      "val_loss"      (default, unchanged) -- lowest weighted cross-entropy.
      "val_macro_f1"  -- highest macro-averaged F1 over the classes.

    Why the option exists: validation loss on this dataset is spiky (occasional jumps
    to 3-4, characteristic of fp16 overflow in the AMP path), so its minimum often
    lands in the first few epochs on a model that has barely learned anything. On the
    618-subject four-way task all three architectures early-stopped with their best
    epoch at 2-5, and EfficientNet-B0 restored an epoch-2 checkpoint that never
    predicted LMCI at all -- 0% recall on a whole class -- while still scoring near the
    majority baseline because it guessed the largest class.

    Macro F1 is the better selector for that task because it averages the classes
    equally, so a checkpoint that has quietly stopped predicting one of the four stages
    is scored as the failure it is rather than rewarded for playing the base rates.

    initial_best_val_loss lets a second call (see train_two_phase below) know about
    a better checkpoint an earlier call already saved, so it won't overwrite it with
    a worse one just because its own tracking restarted from scratch.

    weight_decay adds L2 regularization (penalizes large weights) -- helps prevent
    overfitting when fine-tuning. freeze_bn keeps BatchNorm layers fixed during
    training; see models.freeze_batchnorm for why that matters on a small dataset.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = GradScaler(device="cuda", enabled=(device.type == "cuda"))

    if select_by not in ("val_loss", "val_macro_f1"):
        raise ValueError(f"select_by must be val_loss or val_macro_f1, got {select_by!r}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_macro_f1": []}
    best_val_loss = initial_best_val_loss
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss, train_acc, _, _ = _run_epoch(model, train_loader, criterion, device,
                                                 optimizer, scaler, freeze_bn)
        val_loss, val_acc, val_true, val_pred = _run_epoch(model, val_loader, criterion, device)
        val_macro_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(float(val_macro_f1))
        print(f"Epoch {epoch+1:2d}/{epochs} - train_loss {train_loss:.4f} train_acc {train_acc:.4f}"
              f" - val_loss {val_loss:.4f} val_acc {val_acc:.4f} val_macro_f1 {val_macro_f1:.4f}")

        if select_by == "val_loss":
            improved = val_loss < best_val_loss
        else:
            improved = val_macro_f1 > best_macro_f1

        if improved:
            best_val_loss = min(val_loss, best_val_loss)
            best_macro_f1 = max(val_macro_f1, best_macro_f1)
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no {select_by} improvement for {patience} epochs)")
                break

    print(f"restoring best checkpoint from epoch {best_epoch} (selected by {select_by})")
    history["best_epoch"] = best_epoch
    history["select_by"] = select_by
    model.load_state_dict(torch.load(checkpoint_path))
    return history


def train_two_phase(model, train_loader, val_loader, class_weights, device, checkpoint_path,
                     freeze_epochs=5, freeze_lr=1e-3, unfreeze_epochs=25, unfreeze_lr=1e-5,
                     patience=5, unfreeze_fraction=0.3, weight_decay=1e-4):
    """
    Two-phase fine-tuning for a pretrained model (MobileNetV2 / EfficientNet-B0), tuned to
    avoid the overfitting we saw from unfreezing the entire backbone on a small dataset:

    Phase 1 -- freeze the pretrained backbone, train only the new classifier head for a
    fixed, short number of epochs at a normal learning rate. The head starts as random
    weights, so this quickly gets it into a reasonable range without risking the backbone's
    pretrained features.

    Phase 2 -- unfreeze only the last `unfreeze_fraction` of the backbone's conv/linear
    weights (see models.unfreeze_top_layers) and continue training at a much lower learning
    rate with weight decay and early stopping. Fewer trainable weights + L2 regularization
    target the overfitting the fully-unfrozen version showed (train accuracy climbing to 74%
    while validation accuracy stalled around 53%).

    BatchNorm statistics are deliberately left UNFROZEN (able to update their running
    mean/var), even in the frozen layers -- tried freezing them first and accuracy got
    worse, not better. Our input is grayscale MRI duplicated into 3 "RGB" channels, a very
    different distribution from the natural photos those BatchNorm statistics were computed
    on; forcing the network to keep normalizing with ImageNet-photo statistics instead of
    letting it recalibrate to our actual data hurt every layer's effective features, not just
    the ones being fine-tuned. BN-freezing is standard when the fine-tuning domain is close to
    the original one -- it isn't here.
    """
    from models import freeze_backbone, unfreeze_top_layers

    print("=== Phase 1: training classifier head (backbone frozen) ===")
    freeze_backbone(model)
    history1 = train_model(model, train_loader, val_loader, class_weights, device,
                            epochs=freeze_epochs, lr=freeze_lr, patience=freeze_epochs,
                            checkpoint_path=checkpoint_path)

    print(f"\n=== Phase 2: fine-tuning last {unfreeze_fraction:.0%} of backbone (BN stats adapting, weight_decay={weight_decay}) ===")
    unfreeze_top_layers(model, fraction=unfreeze_fraction)
    history2 = train_model(model, train_loader, val_loader, class_weights, device,
                            epochs=unfreeze_epochs, lr=unfreeze_lr, patience=patience,
                            checkpoint_path=checkpoint_path, initial_best_val_loss=min(history1["val_loss"]),
                            weight_decay=weight_decay)

    history = {k: history1[k] + history2[k] for k in history1}
    history["phase1_epochs"] = len(history1["train_loss"])  # lets plots mark where phase 2 starts
    return history
