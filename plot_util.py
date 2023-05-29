import os

from matplotlib import pyplot as plt


def plot_per_batch_train(train_history, dst_dir):
    train_loss = [batch["loss"] for batch in train_history]
    train_ang_loss = [batch["ang_loss"] for batch in train_history]

    plt.figure()

    plt.plot(train_loss, label='train loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Batches')
    plt.legend()

    plt.savefig(os.path.join(dst_dir, 'training_loss.png'))

    plt.figure()

    plt.plot(train_ang_loss, label='train loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Angular Loss Over Batches')
    plt.legend()

    plt.savefig(os.path.join(dst_dir, 'training_ang_loss.png'))

def plot_per_batch_valid(valid_history, dst_dir):
    valid_loss = [batch["loss"] for batch in valid_history]
    valid_ang_loss = [batch["ang_loss"] for batch in valid_history]

    plt.figure()

    plt.plot(valid_loss, label='validation loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Batches')
    plt.legend()

    plt.savefig(os.path.join(dst_dir, 'validation_loss.png'))

    plt.figure()

    plt.plot(valid_ang_loss, label='validation loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Validation Angular Loss Over Batches')
    plt.legend()

    plt.savefig(os.path.join(dst_dir, 'validation_ang_loss.png'))


def plot_per_epoch(epoch_history, dst_dir):
    train_loss = [epoch["train_loss"] for epoch in epoch_history]
    train_ang_loss = [epoch["train_ang_loss"] for epoch in epoch_history]
    valid_loss = [epoch["valid_loss"] for epoch in epoch_history]
    valid_ang_loss = [epoch["valid_ang_loss"] for epoch in epoch_history]

    plt.figure()

    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.savefig(os.path.join(dst_dir, 'epoch_losses.png'))

    plt.figure()

    plt.plot(train_ang_loss, label='train loss')
    plt.plot(valid_ang_loss, label='valid loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Angular Loss Over Epochs')
    plt.legend()

    plt.savefig(os.path.join(dst_dir, 'epoch_ang_losses.png'))



