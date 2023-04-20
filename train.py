from myModel import *
import torch
from torch import optim
import torch.nn.functional as F
from loadDataset import trainDataset, testDataset
from torch.utils.data import dataloader
import os


def train(device = "cuda:0"):
    epochs = 200
    learning_rate = 0.001
    batchsize = 64

    trainDataLoader = dataloader.DataLoader(
        dataset=trainDataset,
        batch_size=batchsize,
        shuffle=True
    )

    loss_func = F.nll_loss
    model = CNNModel().to(device)
    optimizer_Adam = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for e in range(epochs):
        total_loss = 0
        total_correct_num = 0
        for x, y in trainDataLoader:
            y_pre = model(x.to(device))

            # compute accuracy and loss
            correct_num = (y_pre.argmax(dim=1) == y.to(device)).sum()
            loss = loss_func(y_pre, y.to(device))

            # backward and optimize
            loss.backward()
            optimizer_Adam.step()
            optimizer_Adam.zero_grad()

            # print training info
        print(
            f"Epoch [{e + 1}/{epochs}] Loss: {loss.item():.4f} Acc: {acc:.4f}")

        # break after first batch for testing purposes
        break


def test(model_name, device='cuda:0'):
    batchsize = 64
    testDataLoader = dataloader.DataLoader(
        dataset=testDataset,
        batch_size=batchsize,
        shuffle=False
    )

    model = torch.load(model_name)

    # set the model to evaluation mode
    model.eval()

    # disable gradient calculation
    with torch.no_grad():
        # iterate over the evaluation data loader
        for inputs, targets in testDataLoader:
            # move the data to the device (GPU or CPU)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward pass
            outputs = model(inputs)

            # compute the loss
            loss = nn.CrossEntropyLoss()(outputs, targets)

            # compute the accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == targets).float().mean()

            # aggregate the metrics
            total_loss += loss.item() * inputs.size(0)
            total_acc += acc.item() * inputs.size(0)

        # compute the final evaluation metrics
        avg_loss = total_loss / len(dataset)
        avg_acc = total_acc / len(dataset)

    # print or log the evaluation metrics
    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Validation accuracy: {avg_acc:.4f}")




