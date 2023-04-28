import os


def init_save_files():
    # 保存模型信息的文件
    if not os.path.exists('results/models_info.csv'):
        with open('results/models_info.csv', 'w', encoding='utf-8') as f:
            f.write("model_id,model,epoch,learning_rate,batch_size,optim,loss,acc\n")
    # 保存
    if not os.path.exists('results/results_info.csv'):
        with open('results/results_info.csv', 'w', encoding='utf-8') as f:
            f.write('model_id,loss,acc')


def save_model_info(model_id, **kwargs):
    with open('results/models_info.csv', 'a', encoding='utf-8') as mf:
        mf.write(f"{model_id},{kwargs['model_type']},{kwargs['epoch']},{kwargs['learning_rate']},{kwargs['batch_size']},{kwargs['optim']},{kwargs['loss']},{kwargs['acc']}\n")


def save_test_info(model_id, **kwargs):
    with open('results/results_info.csv', 'a', encoding='utf-8') as tf:
        tf.write(f"{model_id},{kwargs['loss']}, {kwargs['acc']}\n")
