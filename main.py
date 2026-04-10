import torch
import torch.nn as nn
import torch.optim as optim

from models.mobilenet import get_model
from utils.train import train_one_epoch
from utils.evaluate import evaluate
from utils.dataset import get_dataloaders
from configs.config import CONFIG
from domains.base_domain import BaseDomain
from utils.plot import plot_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# domain
def get_domain():
    return BaseDomain()

domain = get_domain()

# data
train_loader, val_loader, test_loader = get_dataloaders(domain)

# model
model = get_model(CONFIG["num_classes"]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

train_acc_list = []
val_acc_list = []

for epoch in range(CONFIG["epochs"]):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_acc = evaluate(model, val_loader, device)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")


plot_accuracy(train_acc_list, val_acc_list)
# test
test_acc = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}")

import json

results = {
    "train_acc": train_acc_list,
    "val_acc": val_acc_list,
    "test_acc": test_acc
}

with open("results_spatial.json", "w") as f:
    json.dump(results, f)

# save
torch.save(model.state_dict(), "mobilenet_spatial.pth")