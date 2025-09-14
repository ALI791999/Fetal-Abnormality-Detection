import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import configs
from data_load import HC18Data
from model import SegModel
from utils import inference_after_epoch
from learnable_resizer import LearnableResizer

import torch.nn.functional as F

def pad_collate(batch, mode="reflect"):
    xs, ys = zip(*batch)
    H = max(x.shape[-2] for x in xs)
    W = max(x.shape[-1] for x in xs)
    xs_pad = [F.pad(x, (0, W - x.shape[-1], 0, H - x.shape[-2]), mode=mode) for x in xs]
    return torch.stack(xs_pad, 0), torch.stack(ys, 0)


def train_model(model,
                resizer,
                train_loader,
                val_loader,
                criterion,
                num_epochs=50,
                patience=5,
                lr=1e-3):

    device = configs.DEVICE
    print(f"Using device: {device}")
    model.to(device)
    resizer.to(device)

    optimizer = Adam(list(model.parameters()) + list(resizer.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        patience=configs.LR_PATIENCE, factor=configs.LR_FACTOR, verbose=True
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # -------- Training --------
        model.train(); resizer.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(resizer(inputs))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        # -------- Validation --------
        model.eval(); resizer.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(resizer(inputs))
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        inference_after_epoch(val_loader, nn.Sequential(resizer, model), epoch, device=device)
        scheduler.step(val_loss)

        # --- save latest (separate folder) + best ---
        ckpt = {"model": model.state_dict(), "resizer": resizer.state_dict()}
        best_path = configs.BEST_MODEL_SAVE_PATH
        latest_path = configs.LATEST_MODEL_SAVE_PATH
        
        base_name = os.path.basename(best_path)
        os.makedirs(os.path.dirname(best_path), exist_ok=True)

        torch.save(ckpt, latest_path)  # always save latest

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(ckpt, best_path)  # save best
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":

    train_dataset = HC18Data(data_type='train')
    val_dataset = HC18Data(data_type='validation')
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, 
                              batch_size=configs.BATCH_SIZE, 
                            #   collate_fn=pad_collate,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=configs.BATCH_SIZE, 
                            # collate_fn=pad_collate,
                            shuffle=False)

    criterion = nn.CrossEntropyLoss()

    model = SegModel().to(configs.DEVICE).train()

    resizer = LearnableResizer(
        in_ch=configs.MODEL_INPUT_CHANNELS,
        out_ch=configs.MODEL_INPUT_CHANNELS,
        out_size=(configs.IMAGE_SIZE, configs.IMAGE_SIZE)
    )

    train_model(model=model,
                resizer=resizer,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                num_epochs=configs.NUM_EPOCHS,
                patience=configs.PATIENCE,
                lr=configs.LEARNING_RATE)
