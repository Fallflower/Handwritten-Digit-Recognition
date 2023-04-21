from myModel import *
import torch
from torch import optim
import torch.nn.functional as F
from loadDataset import trainDataset, testDataset
from torch.utils.data import dataloader
import os


def train(device = "cuda:0"):
    epochs = 20
    learning_rate = 0.001
    batchsize = 1024

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
        total_num = 0
        for batch_id, (x, y) in enumerate(trainDataLoader):
            y_pre = model(x.to(device))

            # compute accuracy and loss

            correct_num = (y_pre.argmax(dim=1) == y.to(device)).sum()
            total_correct_num += correct_num
            # print(correct_num)
            loss = loss_func(y_pre, y.to(device))
            total_loss += loss.item() * batchsize
            # print(loss.item())

            total_num += batchsize
            # backward and optimize
            loss.backward()
            optimizer_Adam.step()
            optimizer_Adam.zero_grad()

            # break
        acc = total_correct_num / total_num
        Loss = total_loss / total_num
        print(
            f"Epoch [{e + 1}/{epochs}] Loss: {Loss:.4f} Acc: {acc:.4f}")

        # break after first batch for testing purposes
        # break
    torch.save(model, 'models/1.pt')

def test(model_name, device='cuda:0'):
    batchsize = 64
    testDataLoader = dataloader.DataLoader(
        dataset=testDataset,
        batch_size=batchsize,
        shuffle=False
    )

    model = torch.load(model_name)
    model.to(device)

    # set the model to evaluation mode
    model.eval()

    loss_func = F.nll_loss

    # disable gradient calculation
    with torch.no_grad():
        # iterate over the evaluation data loader
        total_loss = 0
        total_correct_num = 0
        total_num = 0
        for x, y in testDataLoader:
            # forward pass
            y_pre = model(x.to(device))

            # compute the loss
            loss = loss_func(y_pre, y.to(device))

            # compute the accuracy
            preds = torch.argmax(y_pre, dim=1)
            correct_num = (preds == y.to(device)).sum()

            # aggregate the metrics
            total_loss += loss.item() * batchsize
            total_correct_num += correct_num
            total_num += batchsize

        # compute the final evaluation metrics
        loss = total_loss / total_num
        acc = total_correct_num / total_num

    # print or log the evaluation metrics
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc:.4f}")




