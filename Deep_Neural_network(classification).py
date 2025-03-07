from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
dataset=pd.read_csv("C:/Users/Vinish/Downloads/diabetes.csv",header=0)
print(dataset)
X=dataset.iloc[:,0:8].values
Y=dataset.iloc[:,8].values
print(X)
print(Y)
model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
model.fit(X,Y,epochs=10,batch_size=10)
_,accuracy=model.evaluate(X,Y)
print("Accuracy of Model is",(accuracy*100))
prediction=model.predict_step(X)
for i in range(5):
    print(X[i].tolist(),prediction[i],Y[i])
