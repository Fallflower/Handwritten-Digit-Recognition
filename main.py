from train import train, test
from saveModelInfo import init_save_files
from myModel import CNNModel, AttentionModel

if __name__ == '__main__':
    init_save_files()
    model_id = train(set_model=CNNModel,
                     epochs=40,
                     learning_rate=0.0008,
                     batchsize=256)
    test(model_id)
