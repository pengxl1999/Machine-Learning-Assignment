import numpy as np

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

batch_size = 50
epochs = 4000
loaded_data = datasets.load_boston()
data_x = loaded_data.data
data_y = loaded_data.target
# print(data_X[:2, :])
# print(data_y[:2])
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
x_scaler = StandardScaler()
x_scaler.fit(x_train)
x_train_standard = x_scaler.transform(x_train)
x_test_standard = x_scaler.transform(x_test)
x_bgd_train = np.c_[x_train_standard, np.ones((x_train_standard.shape[0], 1))]
x_bgd_test = np.c_[x_test_standard, np.ones((x_test_standard.shape[0], 1))]

class BGDRegressor:
    def bgd(self, xx, yy, theta, learning_rate, m):
        bgd_curve_x = []
        bgd_curve_y = []
        data_generator = get_batch(xx, yy)
        x, y = next(data_generator)
        x_trans = x.transpose()
        for i in range(epochs):
            hy = np.dot(x, theta)
            loss = hy - y
            gradient = np.dot(x_trans, loss) / m
            theta = theta - learning_rate * gradient
            #print(r2_score(y_test.reshape(-1, 1), self.predict(x_bgd_test, theta), multioutput='variance_weighted'))
            #print(loss[0])
            bgd_curve_x.append(i)
            bgd_curve_y.append(r2_score(y_test.reshape(-1, 1), self.predict(x_bgd_test, theta),
                                        multioutput='variance_weighted'))
        return theta, bgd_curve_x, bgd_curve_y

    def predict(self, x, theta):
        y = np.dot(x, theta)
        # print(y)
        return y


def get_batch(x, y):
    k = 0
    while True:
        if k + batch_size < len(x):
            yield x[k: k + batch_size], y[k: k + batch_size]
            k += batch_size
        else:
            yield x[k: len(x) - 1], y[k: len(x) - 1]
            k = 0


def SGDTrain():
    model = SGDRegressor()
    # model.fit(x_train_standard, y_train)
    # print(model.coef_)
    # print(model.intercept_)
    data_generator = get_batch(x_train_standard, y_train)
    sgd_curve_x = []
    sgd_curve_y = []
    for i in range(epochs):  # Train for 100 epochs
        x, y = next(data_generator)
        # print(x)
        # print(y)
        model.partial_fit(x, y)
        # print(model.score(x_test_standard, y_test))
        # print(model.coef_)
        sgd_curve_x.append(i)
        sgd_curve_y.append(model.score(x_test_standard, y_test))
    predicted = model.predict(x_test_standard)
    plt.title('SGD result (4000 epochs)')
    plt.scatter(y_test, predicted, color='y', marker='o')
    plt.plot(y_test, y_test, color='g')
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    # plt.savefig('./4000_1.png')
    plt.show()
    print('SGD RMSE为：', np.sqrt(mean_squared_error(y_test, predicted)))
    return sgd_curve_x, sgd_curve_y


def BGDTrain():
    model = BGDRegressor()
    theta = np.ones((14, 1))
    learning_rate = 0.01
    theta, bgd_curve_x, bgd_curve_y = model.bgd(x_bgd_train, y_train.reshape(-1, 1), theta, learning_rate, x_bgd_train.shape[0])
    predicted = model.predict(x_bgd_test, theta)
    plt.title('BGD result (4000 epochs)')
    plt.scatter(y_test, predicted, color='y', marker='o')
    plt.plot(y_test, y_test, color='g')
    # plt.scatter(y_test, y_test, color='g', marker='+')
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    # plt.savefig('./4000_2.png')
    plt.show()
    print('BGD RMSE为：', np.sqrt(mean_squared_error(y_test, predicted)))
    return bgd_curve_x, bgd_curve_y


if __name__ == '__main__':
    sgd_curve_x, sgd_curve_y = SGDTrain()
    bgd_curve_x, bgd_curve_y = BGDTrain()
    plt.title('Comparison between SGD and BGD (4000 epochs)')
    plt.plot(sgd_curve_x, sgd_curve_y, color='b', label='SGD')
    plt.plot(bgd_curve_x, bgd_curve_y, color='r', label='BGD')
    plt.xlabel('Epochs')
    plt.ylabel('The coefficient of determination R^2 of the prediction')
    plt.ylim(-1, 1)
    plt.legend()
    # plt.savefig('./4000_3.png')
    plt.show()
