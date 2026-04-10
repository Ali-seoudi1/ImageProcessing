from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from configs.config import CONFIG

def get_dataloaders(domain):

    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    # Apply domain transform
    train_data.dataset.transform = transform
    val_data.dataset.transform = transform
    test_data.dataset.transform = transform

    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_data, batch_size=CONFIG["batch_size"])

    return train_loader, val_loader, test_loader