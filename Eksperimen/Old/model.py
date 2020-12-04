from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential       
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
import tensorflow.keras
tensorflow.keras.backend.set_floatx('float64')

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def neuralNet(neurons, drop, lr, layer, **kwargs):
    tensorflow.keras.backend.clear_session()
    neurons = int(neurons)
    layer = int(layer)
    opt = optimizers.Adam(learning_rate=lr)
    model = Sequential()
    model.add(Dense(neurons, activation='relu', kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=42)))
    model.add(Dropout(drop))
    for x in range(layer - 1):
        model.add(Dense(neurons, activation='relu', kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=42)));
        model.add(Dropout(drop))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def random_forest(depth, estimator, **kwargs):
    depth = int(depth)
    estimator = int(estimator)
    model = RandomForestClassifier(random_state=42, max_depth=depth, n_estimators=estimator)
    return model

def decission_tree(depth, **kwargs):
    depth = int(depth)
    model = DecisionTreeClassifier(max_depth= depth)
    return model

def naive_bayes(**kwargs):
    model = GaussianNB()
    return model

def Logistic_regression(c, **kwargs):
    model = LogisticRegression(C=c) 
    return model

def knn(leaf,n, **kwargs):
    leaf = int(leaf)
    n = int(n)
    model = KNeighborsClassifier(leaf_size=leaf, n_neighbors=n)
    return model