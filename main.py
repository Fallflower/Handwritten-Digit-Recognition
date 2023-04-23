from train import train, test
from myModel import CNNModel, AttentionModel


if __name__ == '__main__':
    train(set_model=CNNModel,
          epochs=30,
          learning_rate=0.001,
          batchsize=1024)
    test('models/2.pt')
