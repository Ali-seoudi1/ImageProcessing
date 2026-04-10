import matplotlib.pyplot as plt

def plot_accuracy(train_acc, val_acc):
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.savefig("accuracy_plot.png")
    plt.show()