import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from .data import make_loaders
from .model import CifarResNet

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = correct = count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return {"loss": total_loss / count, "accuracy": correct / count}

def main():
    parser = argparse.ArgumentParser(description="Train a residual CNN on CIFAR-10.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers cannot be negative")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train, validation, test, classes = make_loaders(args.data_dir, args.batch_size, args.workers, args.seed)
    model = CifarResNet(len(classes)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train(); total_loss = count = 0
            for images, labels in train:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(images), labels)
                loss.backward(); optimizer.step()
                total_loss += loss.item() * labels.size(0); count += labels.size(0)
            scheduler.step()
            val = evaluate(model, validation, device, loss_fn)
            record = {"epoch": epoch, "train_loss": total_loss / count, "validation": val}
            print(json.dumps(record)); log.write(json.dumps(record) + "\n"); log.flush()
            if val["accuracy"] > best_accuracy:
                best_accuracy = val["accuracy"]
                torch.save({"model_state": model.cpu().state_dict(), "classes": classes, "seed": args.seed, "epoch": epoch}, output / "best.pt")
                model.to(device)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    result = {"selected_epoch": checkpoint["epoch"], "test": evaluate(model, test, device, loss_fn)}
    (output / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
