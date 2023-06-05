from train import train, test, train_
from saveModelInfo import init_save_files
from myModel import CNNModel, AttentionModel

if __name__ == '__main__':
    init_save_files()
    # model_id = train(set_model=CNNModel,
    #                  epochs=30,
    #                  learning_rate=0.0009,
    #                  batchsize=1024)
    model_id = train_(50, 0.008, 1024)
    test(model_id)
