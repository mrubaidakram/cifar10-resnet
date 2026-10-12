from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

def make_loaders(root: str, batch_size: int, workers: int, seed: int):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    augmented = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    clean = datasets.CIFAR10(root=root, train=True, download=False, transform=eval_transform)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(augmented), generator=generator).tolist()
    validation_size = 5_000
    train_idx, validation_idx = order[validation_size:], order[:validation_size]
    train = Subset(augmented, train_idx)
    validation = Subset(clean, validation_idx)
    options = {"batch_size": batch_size, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    return (DataLoader(train, shuffle=True, **options), DataLoader(validation, shuffle=False, **options),
            DataLoader(test, shuffle=False, **options), augmented.classes)
