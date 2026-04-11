import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from configs.config import CONFIG


class DomainDataset(Dataset):
    def __init__(self, base_dataset, indices, domain, image_transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.domain = domain
        self.image_transform = image_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_index = self.indices[idx]
        image, label = self.base_dataset[image_index]
        transformed_image = self.domain.transform(image)
        return self.image_transform(transformed_image), label

    def get_raw_image(self, idx):
        image_index = self.indices[idx]
        image, label = self.base_dataset[image_index]
        return self.domain.transform(image), label


def _build_base_dataset():
    data_dir = CONFIG["data_dir"]
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)

    class CIFAR10Dataset(Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            image = Image.fromarray(self.data[idx]).convert("RGB")
            return image, self.targets[idx]

    return CIFAR10Dataset(train_dataset.data, train_dataset.targets)


def _split_indices(dataset_length):
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    shuffled_indices = torch.randperm(dataset_length, generator=generator).tolist()

    train_end = int(CONFIG["train_split"] * dataset_length)
    val_end = train_end + int(CONFIG["val_split"] * dataset_length)

    train_indices = shuffled_indices[:train_end]
    val_indices = shuffled_indices[train_end:val_end]
    test_indices = shuffled_indices[val_end:]
    return train_indices, val_indices, test_indices


def _build_image_transform():
    return transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["normalize_mean"], CONFIG["normalize_std"]),
    ])


def get_dataloaders(domain):
    base_dataset = _build_base_dataset()
    train_indices, val_indices, test_indices = _split_indices(len(base_dataset))
    image_transform = _build_image_transform()

    train_data = DomainDataset(base_dataset, train_indices, domain, image_transform)
    val_data = DomainDataset(base_dataset, val_indices, domain, image_transform)
    test_data = DomainDataset(base_dataset, test_indices, domain, image_transform)

    loader_kwargs = {
        "batch_size": CONFIG["batch_size"],
        "num_workers": CONFIG["num_workers"],
    }

    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
