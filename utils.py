import matplotlib.pyplot as plt

def plot_stats(train_losses, valid_losses):

    plt.title("Train losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(train_losses)
    plt.savefig("gpt/images/train.png")
    plt.show()

    plt.title("Valid losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(valid_losses, color="orange")
    plt.savefig("gpt/images/valid.png")
    plt.show()

    plt.title("Losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Valid")
    plt.legend(loc="upper right")
    plt.savefig("gpt/images/both.png")
    plt.show()