import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the data from the given path and returns the X and y dataframes"""
    X = pd.read_csv(path, sep=" ", header=None)
    y, X = X[6].str.replace('*', ''), X.drop([4, 5, 6], axis=1)
    y, X = y.astype(float), X.astype(float)
    return X, y


def read_data2(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the data from the given path and returns the X and y dataframes"""
    X = pd.read_csv(path, sep=",")
    y, X = X['Outcome'], X.drop(['Outcome'], axis=1)
    y = y.apply(lambda x: -1 if x == 0 else x)
    y, X = y.astype(float), X.astype(float)
    return X, y


def send_model_to_AMPL(X: pd.DataFrame,
                       y: pd.DataFrame,
                       nu: list,
                       path: str = "",
                       kernel: bool = False) -> None:
    """Sends the data to AMPL format
      Args:
        X: Dataframe with the features
        y: Dataframe with the labels
        nu: List with the nu values
        path: Path to save the data
    """

    X.index, y.index = range(1, len(X)+1), range(1, len(y)+1)

    nu = pd.DataFrame(nu)
    nu.index += 1

    open(path, 'w').close()
    with open(path, 'a') as f:
        f.write('param m := %d;\n' % len(X))
        f.write('param n := %d;\n' % len(X.columns))
        f.write('param K := %d;\n' % len(nu))

        f.write('param nus := \n')
        nu.to_csv(f, header=False, sep=" ")
        f.write(';\n\n')

        if kernel:
            f.write('param sigma := %d; \n' % len(X.columns))

        f.write('param y := \n')
        y.to_csv(f, header=False, sep=" ")
        f.write(';\n\n')

        text = " "
        for i in range(1, len(X.columns)+1):
            text = text + str(i) + " "
        f.write('param A : \n' + text + ':= \n')
        X.to_csv(f, header=False, sep=" ")
        f.write(';\n\n')


def get_parameters(filename, columns) -> tuple[float, int, float, list, float]:
    """Reads the parameters from the given file"""

    with open(filename) as f:
        lines = f.readlines()
        objective = float(lines[0].split()[-1][:-1])
        iterations = int(lines[1].split()[0])
        time = float(lines[4].split()[2])
        w = lines[7:columns+7]
        w = [float(x.split()[1]) for x in w]
        gamma = float(lines[-2].split()[2])
        return objective, iterations, time, w, gamma


def get_parameters2(filename) -> tuple[float, int, float, float]:
    """Reads the parameters from the given file"""

    with open(filename) as f:
        lines = f.readlines()
        objective = float(lines[0].split()[-1][:-1])
        iterations = int(lines[1].split()[0])
        time = float(lines[4].split()[2])
        gamma = float(lines[-2].split()[2])
        return objective, iterations, time, gamma


def getAccuracy(w, gamma, X, y) -> float:
    """Returns the accuracy of the model"""
    y_pred = np.dot(X, w) + gamma
    y_pred = np.where(y_pred > 0, 1, -1)
    return np.sum(y_pred == y)/len(y)


def send_test_to_AMPL(X, dataname="") -> None:
    """Sends the test data to AMPL format"""
    X = pd.DataFrame(X)
    X.index = range(1, len(X)+1)

    open(dataname, 'w').close()
    with open(dataname, 'a') as f:
        f.write('param TEST_SIZE := %d; \n' % len(X))
        text = " "
        for i in range(1, len(X.columns)+1):
            text = text + str(i) + " "
        f.write('param test : \n' + text + ':= \n')
        X.to_csv(f, header=False, sep=" ")
        f.write(';\n\n')
