import keras
import numpy as np
from keras.models import load_model
from keras.callbacks import Callback
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

loaded_data = datasets.load_boston()
data_x = loaded_data.data
data_y = loaded_data.target
# print(data_X[:2, :])
# print(data_y[:2])
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)


class LossHistory(Callback):  # 继承自Callback类

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 在每一个batch结束后记录相应的值
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    # 在每一个epoch之后记录相应的值
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        # plt.legend(loc="upper right")
        plt.savefig("loss.png")
        plt.show()


if __name__ == '__main__':
    job = input()
    if job == 'train':
        epochs = 2000
        model = keras.models.Sequential([
            keras.layers.Dense(1, input_dim=13)
        ])
        model.summary()
        model.compile(optimizer='adam',
                      loss='mse')
        history = LossHistory()
        model.fit(x_train, y_train, epochs=epochs, callbacks=[history])
        model.evaluate(x_test, y_test)
        history.loss_plot('epoch')
        # model.save('./boston_model_' + str(epochs) + '.h5')
    else:
        model_path = './boston_model_2000.h5'
        load_model = load_model(model_path)
        predicted = load_model.predict(x_test)
        plt.title('Neural network result (2000 epochs)')
        plt.scatter(y_test, predicted, color='y', marker='o')
        plt.plot(y_test, y_test, color='g')
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.savefig('./nn_500.png')
        plt.show()
        print('NN RMSE为：', np.sqrt(mean_squared_error(y_test, predicted)))

