import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from device_util import available_device
from angular_util import angular_loss, pitchyaw2xyz
from plot_util import plot_per_epoch, plot_per_batch_train, plot_per_batch_valid

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(6)
        self.regression = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 2),
        )
        self.device = available_device()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)

        x = torch.flatten(x, 1)

        x = self.regression(x)

        return x

    def train_net(self, epoch):
        self.train()
        train_loss, train_ang_loss = 0, 0

        for X, y in tqdm(self.train_loader, desc=f"(Train) Epoch {epoch} [{self.device}]"):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.forward(X)
            loss = self.l1_loss(pred, y)
            ang_loss = self.ang_loss(
                pitchyaw2xyz(pred),
                pitchyaw2xyz(y)
            )

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_ang_loss += ang_loss.item()

            self.train_step_history.append({
                "loss": loss.item(),
                "ang_loss": ang_loss.item()
            })

        train_loss /= len(self.train_loader)
        train_ang_loss /= len(self.train_loader)

        return train_loss, train_ang_loss


    def eval_net(self, epoch):
        self.eval()

        valid_loss, valid_ang_loss = 0, 0
        with torch.no_grad():
            for X, y in tqdm(self.valid_loader, desc=f"(Valid) Epoch {epoch} [{self.device}]"):
                X, y = X.to(device), y.to(device)
                pred = self.forward(X)
                loss = self.l1_loss(pred, y)
                ang_loss = self.ang_loss(
                    pitchyaw2xyz(pred),
                    pitchyaw2xyz(y)
                )

                valid_loss += loss.item()
                valid_ang_loss += ang_loss.item()

                self.val_step_history.append({
                    "loss": loss.item(),
                    "ang_loss": ang_loss.item()
                })

            valid_loss /= len(self.valid_loader)
            valid_ang_loss /= len(self.valid_loader)

        return valid_loss, valid_ang_loss



    def train_process(self, train_loader, valid_loader, epochs, lr, dst_dir):
        self.l1_loss = F.l1_loss
        self.ang_loss = angular_loss
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        device = available_device()
        self.device = device
        self.to(self.device)

        os.makedirs(dst_dir, exist_ok=True)

        self.train_step_history = []
        self.val_step_history = []
        self.epoch_history = []
        best_loss = None
        out_path = os.path.join(dst_dir, f"output.txt")

        for t in range(epochs):
            train_loss, train_ang_loss = self.train_net(t)
            valid_loss, valid_ang_loss = self.eval_net(t)

            self.epoch_history.append({
                "train_loss": train_loss,
                "train_ang_loss": train_ang_loss,
                "valid_loss": valid_loss,
                "valid_ang_loss": valid_ang_loss
            })

            with open(out_path, "a") as file:
                file.write(f"Epoch {t + 1}:\n")
                file.write(f"train_l1_loss {train_loss:.4f}, train_ma_loss {train_ang_loss:.4f}\n")
                file.write(f"valid_l1_loss {valid_loss:.4f}, valid_ma_loss {valid_ang_loss:.4f}\n")
                file.write("\n")

            if best_loss is None or best_loss > valid_loss:
                best_loss = valid_loss
                torch.save(
                    self.state_dict(),
                    os.path.join(dst_dir, 'weights.pth')
                )

            print(f"Train Loss (L1): {train_loss:.4f}, Train Loss (Mean Angular Loss): {train_ang_loss:.4f}")
            print(f"Valid Loss (L1): {valid_loss:.4f}, Valid Loss (Mean Angular Loss): {valid_ang_loss:.4f}")
            print()

        plot_per_batch_train(self.train_step_history, dst_dir)
        plot_per_batch_valid(self.val_step_history, dst_dir)
        plot_per_epoch(self.epoch_history, dst_dir)





    def test_process(self, test_loader, dst_dir, poisoned):
        self.l1_loss = F.l1_loss
        self.ang_loss = angular_loss
        self.valid_loader = test_loader
        device = available_device()
        self.device = device
        self.to(self.device)

        os.makedirs(dst_dir, exist_ok=True)

        self.val_step_history = []

        file_name = "output.txt"
        if poisoned:
            file_name = "output-p.txt"

        out_path = os.path.join(dst_dir, file_name)

        epochs = 1
        test_loss, test_ang_loss = self.eval_net(epochs)

        with open(out_path, "a") as file:
            file.write(f"test_loss {test_loss:.4f}, test_ang_loss {test_ang_loss:.4f}\n")
            file.write("\n")

        print(f"Test Loss (L1): {test_loss:.4f}, Test Loss (Mean Angular Loss): {test_ang_loss:.4f}")






















