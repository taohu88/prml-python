import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def create_toy_data(func, sample_sz, std):
    x = np.linspace(0, 1, sample_sz)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


def sin_x(x):
    return np.sin(2*np.pi*x)


if __name__ == "__main__":
    np.random.seed(1234)

    sample_sz = 10
    test_sz = 100
    x_train, y_train = create_toy_data(sin_x, sample_sz, 0.25)
    x_test = np.linspace(0, 1, test_sz)
    y_test = sin_x(x_test)

    # plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
    # plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    # plt.legend()
    # plt.show()

    x_train = x_train[:, np.newaxis]
    x_test = x_test[:, np.newaxis]

    for i, degree in enumerate([0, 1, 3, 9]):
        plt.subplot(2, 2, i + 1)
        feature = PolynomialFeatures(degree)
        X_train = feature.fit_transform(x_train)
        X_test = feature.fit_transform(x_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        print(model.coef_)
        y = model.predict(X_test)

        plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
        plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
        plt.plot(x_test, y, c="r", label="fitting")
        plt.ylim(-1.5, 1.5)
        plt.annotate("M={}".format(degree), xy=(-0.15, 1))

    plt.legend(bbox_to_anchor=(1.05, 0.64), loc=2, borderaxespad=0.)
    plt.show()