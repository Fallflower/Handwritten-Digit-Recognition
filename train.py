from myModel import *
from saveModelInfo import *
import torch
import os
from torch import optim
import torch.nn.functional as F
from loadDataset import trainDataset, testDataset
from torch.utils.data import dataloader
import os
from transformer import VisionTransformer


def get_config():
    config_dict = {
        "hidden_size": 128,
        'patch_size': 14,
        "mlp_dim": 1024,
        'num_heads': 8,
        'num_layers': 4,
        'attention_dropout_rate': 0.5,
        'dropout_rate': 0.1,
        'classifier': 'token',
        'representation_size': None
    }
    return config_dict


def train(set_model, epochs, learning_rate, batchsize, optimizer=optim.Adam, device="cuda:0"):

    trainDataLoader = dataloader.DataLoader(
        dataset=trainDataset,
        batch_size=batchsize,
        shuffle=True
    )

    loss_func = F.nll_loss
    model = set_model().to(device)
    optimizer_Adam = optimizer(model.parameters(), lr=learning_rate)

    mid = len(os.listdir('models')) + 1
    mf_dic = {
        'model_type': None,
        'epoch': None,
        'learning_rate': learning_rate,
        'batch_size': batchsize,
        'optim': 'Adam',
        'loss': None,
        'acc': None
    }
    if set_model == CNNModel:
        mf_dic['model_type'] = 'CNNModel'
    elif set_model == AttentionModel:
        mf_dic['model_type'] = 'AttentionModel'

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
        mf_dic['epoch']=e;mf_dic['loss']=Loss;mf_dic['acc']=acc
        save_model_info(mid, **mf_dic)

    torch.save(model, 'models/%d.pt' % mid)
    return mid


def train_(epochs, learning_rate, batch_size, device="cuda:0"):

    trainDataLoader = dataloader.DataLoader(
        dataset=trainDataset,
        batch_size=batch_size,
        shuffle=True
    )
    config = get_config()
    model = VisionTransformer(**config).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    mid = len(os.listdir('models')) + 1
    mf_dic = {
        'epoch': None,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'optim': 'Adam',
        'loss': None,
        'acc': None,
        'model_type': 'AttentionModel'
    }
    loss_func = nn.CrossEntropyLoss()

    model.train()
    for e in range(epochs):
        total_loss = 0.0
        total_correct_num = 0
        total_num = 0
        for batch_id, (x, y) in enumerate(trainDataLoader):
            y_pre = model(x.to(device))

            preds = torch.argmax(y_pre, dim=1)

            correct_num = (preds==y.to(device)).sum()

            loss = loss_func(y_pre, y.to(device))
            # print(loss.item())
            total_num += y.shape[0]
            total_loss += loss.item()
            total_correct_num += correct_num
            # backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        acc = total_correct_num / total_num
        Loss = total_loss / total_num
        print(
            f"Epoch [{e + 1}/{epochs}] Loss: {Loss:.6f} Acc: {acc:.4f}")
        mf_dic['epoch']=e;mf_dic['loss']=Loss;mf_dic['acc']=acc
        save_model_info(mid, **mf_dic)

    torch.save(model, 'models/%d.pt' % mid)
    return mid


def test(model_id: int, device='cuda:0'):
    batchsize = 1024
    testDataLoader = dataloader.DataLoader(
        dataset=testDataset,
        batch_size=batchsize,
        shuffle=False
    )

    model = torch.load('models/%d.pt' % model_id)
    model.to(device)

    # set the model to evaluation mode
    model.eval()

    loss_func = nn.CrossEntropyLoss()

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
            total_loss += loss.item()
            total_correct_num += correct_num
            total_num += y.shape[0]

        # compute the final evaluation metrics
        loss = total_loss / total_num
        acc = total_correct_num / total_num

    tf_dic = {
        'loss': loss,
        'acc': acc
    }

    # print or log the evaluation metrics
    print(f"Validation loss: {loss:.6f}")
    print(f"Validation accuracy: {acc:.4f}")

    save_test_info(model_id, **tf_dic)
