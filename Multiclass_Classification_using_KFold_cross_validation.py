#code 1
'''
import pandas
from keras.models import Sequential
from keras.layers import Dense,Input
from scikeras.wrappers import KerasClassifier
from keras import utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
df=pandas.read_csv("C:/Users/Vinish/OneDrive/Desktop/DL/flowers.csv",header=None)
print(df)
X = df.iloc[:,0:4].astype(float)
y=df.iloc[:,4]
encoder=LabelEncoder()
encoder.fit(y)
encoded_y=encoder.transform(y)
print(encoded_y)
dummy_Y=utils.to_categorical(encoded_y)
print(dummy_Y)
def baseline_model():
    model = Sequential()
    model.add(Input(shape=(4,))) 
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator=baseline_model()
estimator.fit(X,dummy_Y,epochs=100,shuffle=True)
action=estimator.predict(X)
for i in range(25):
    print(dummy_Y[i])
print('^^^^^^^^^^^^^^^^^^^^^^')
for i in range(25):
    print(action[i])

'''
#code 2:
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from scikeras.wrappers import KerasClassifier
from keras import utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv("C:/Users/Vinish/OneDrive/Desktop/DL/flowers.csv", header=None)
dataset1 = dataset.values
X = dataset1[:, 0:4].astype(float)
Y = dataset1[:, 4]
encoder = LabelEncoder()
encoder.fit(Y)
encoder_Y = encoder.transform(Y)
print(encoder_Y)
dummy_Y = utils.to_categorical(encoder_Y)
print(dummy_Y)
def baseline_model():
    model = Sequential()
    model.add(Input(shape=(4,)))
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(
    model=baseline_model,
    epochs=100,
    batch_size=5,
    verbose=0
    
)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
