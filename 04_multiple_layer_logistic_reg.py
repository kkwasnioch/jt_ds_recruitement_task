import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class MultilayerLogReg(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st layer
            torch.nn.Linear(num_features, 150),
            torch.nn.ReLU(),
            # 2nd layer
            torch.nn.Linear(150, 75),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(75, num_classes),
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return self.labels.shape[0]


def compute_eval(model, dataloader):
    model = model.eval()

    correct = 0.0
    total_examples = 0
    for idx, (features, labels) in enumerate(dataloader):
        with torch.inference_mode():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)

        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct/total_examples


if __name__ == '__main__':

    df = pd.read_csv('to_model_train.csv', index_col=0)

    y = df['code'].values
    X = df.drop(['code', 'uid', 'keywords', 'content_url', 'content'], axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=1,
                                                        stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.1,
                                                      random_state=1,
                                                      stratify=y_train)

    print(X_train.shape)
    print(X_test.shape)
    print(X_val.shape)
    # Plot only if feature shape is 2D
    # plt.plot(
    #     X_train[y_train == 0, 0],
    #     X_train[y_train == 0, 1],
    #     marker="D",
    #     markersize=10,
    #     linestyle="",
    #     label="Class 0",
    # )
    #
    # plt.plot(
    #     X_train[y_train == 1, 0],
    #     X_train[y_train == 1, 1],
    #     marker="^",
    #     markersize=13,
    #     linestyle="",
    #     label="Class 1",
    # )
    #
    # plt.legend(loc=2)
    #
    # plt.xlim([-5, 5])
    # plt.ylim([-5, 5])
    #
    # plt.xlabel("Feature $x_1$", fontsize=12)
    # plt.ylabel("Feature $x_2$", fontsize=12)

    # plt.grid()
    # plt.show()

    train_ds = MyDataset(X_train, y_train)
    test_ds = MyDataset(X_test, y_test)
    val_ds = MyDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(dataset=val_ds, batch_size=32, shuffle=False)

    # Training loop

    torch.manual_seed(1)

    model = MultilayerLogReg(num_features=300, num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    num_epochs = 10

    for epoch in range(num_epochs):

        model = model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):

            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if not batch_idx % 20:
                print(f"Epoch: {epoch + 1}/{num_epochs + 1}"
                      f"| Batch: {batch_idx}/{len(train_loader)}"
                      f"| Loss: {loss}")
            train_acc = compute_eval(model, train_loader)
            val_acc  = compute_eval(model, val_loader)
            print(f"Train acc: {train_acc*100}%, val acc: {val_acc*100}%")
    # Evaluation
    train_acc = compute_eval(model, train_loader)
    val_acc  = compute_eval(model, val_loader)
    test_acc = compute_eval(model, test_loader)

    print(f"Train acc: {train_acc*100}%, "
          f"|val acc: {val_acc*100}%, "
          f"|test acc: {test_acc*100}%")