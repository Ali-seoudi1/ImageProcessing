import torch


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_examples = 0
    predictions = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total_examples += batch_size

            predictions.extend(preds.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

    average_loss = total_loss / total_examples
    accuracy = correct / total_examples
    return average_loss, accuracy, predictions, labels_list
