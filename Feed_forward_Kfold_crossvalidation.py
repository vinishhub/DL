#pip install fsspec
#pip install scikit-learn==1.5.2
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
dataframe = pd.read_csv("C:/Users/Vinish/OneDrive/Desktop/DL/housing.csv",sep='\s+', header=None)
dataset = dataframe.values
X = dataset[:, 0:13]
Y = dataset[:, 13]
def wider_model():
    inputs = Input(shape=(13,))
    x = Dense(15, kernel_initializer='normal', activation='relu')(inputs)
    #x = Dense(20, kernel_initializer='normal', activation='relu')(inputs) Modifying Neurons
    x = Dense(13, kernel_initializer='normal', activation='relu')(x)
    outputs = Dense(1, kernel_initializer='normal')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=wider_model, epochs=10, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))


