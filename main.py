import torch
import torch.nn as nn
import torch.optim as optim

from models.mobilenet import get_model
from utils.train import train_one_epoch
from utils.evaluate import evaluate
from utils.dataset import get_dataloaders
from configs.config import CONFIG
from domains.base_domain import BaseDomain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# domain
domain = BaseDomain()

# data
train_loader, val_loader, test_loader = get_dataloaders(domain)

# model
model = get_model(CONFIG["num_classes"]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

for epoch in range(CONFIG["epochs"]):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_acc = evaluate(model, val_loader, device)

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

# test
test_acc = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}")

# save
torch.save(model.state_dict(), "mobilenet_spatial.pth")