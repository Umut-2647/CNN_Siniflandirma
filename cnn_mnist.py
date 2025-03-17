from keras.models import Sequential #type:ignore
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization #type:ignore
from keras.utils import to_categorical #type:ignore
import matplotlib.pyplot as plt 
from glob import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def load_and_preprocess(data_path):
    data= pd.read_csv(data_path)
    data = data.to_numpy() #arraye çeviriyoruz
    np.random.shuffle(data) #datayi kariştiriyoruz
    x = data[:, 1:].reshape(-1, 28, 28, 1) / 255.0 #datayi 28x28x1 e çeviriyoruz ve normalize ediyoruz (resim kismi)
    y = data[:, 0].astype(np.int32) #labeli aliyoruz ve int e çeviriyoruz
    y = to_categorical(y, num_classes=len(set(y))) #one hot encoding yapiyoruz

    return x,y

train_data_path= "mnist_data/mnist_train.csv"
test_data_path= "mnist_data/mnist_test.csv"

x_train,y_train = load_and_preprocess(train_data_path) 
x_test,y_test = load_and_preprocess(test_data_path)

print("x_train shape: ",x_train.shape)
print("x_test shape: ",x_test.shape)

#%%
index=3 #rastgele bir sayi ile gorsellestirme
vis= x_train.reshape(60000,28,28) #datayi 28x28 e çeviriyoruz cunku gorsellestirme icin sondaki 1 kanala ihtiyacimiz yok
plt.imshow(vis[index])  #gorsellestirme
plt.legend()
plt.axis("off")
plt.show()
print(np.argmax(y_train[index])) #labeli yazdiriyoruz


#%% CNN Model

numberOfClasses= y_train.shape[1] #label sayisi (60000,10)
model=Sequential()

model.add(Conv2D(input_shape=(28,28,1),filters=32,kernel_size=(3,3))) #input_shape veriyoruz
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64,kernel_size=(3,3))) #ikinci conv layer ekliyoruz
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters=128,kernel_size=(3,3))) 
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten()) #flatten layer ekliyoruz
model.add(Dense(256)) #256 nöronlu bir layer ekliyoruz
model.add(Activation("relu"))
model.add(Dropout(0.2)) 
model.add(Dense(numberOfClasses)) #output layer ekliyoruz
model.add(Activation("softmax")) 

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

hist= model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=8,batch_size=4000)
 
model.save("mnist_json_agirlik/cnn_mnist.h5") #modeli kaydediyoruz

#%%
print(hist.history.keys()) #hist parametresi neleri tutuyor onu gösteriyoruz
#hist parametresi sunlari tutyor: loss, accuracy, val_loss, val_accuracy
plt.plot(hist.history["loss"],label="Train loss")
plt.plot(hist.history["val_loss"],label="Validation loss")
plt.legend()
plt.show()

plt.figure()

plt.plot(hist.history["accuracy"],label="Train accuracy")
plt.plot(hist.history["val_accuracy"],label="Validation accuracy")
plt.legend()
plt.show()

#%% Save history     ##burda kaydetme onemli cuz her seferinde modeli tekrar çaliştirmak yerine sadece history i yükleyerek görebiliriz
import json
with open("mnist_json_agirlik/cnn_mnist.json","w") as f:
    json.dump(hist.history,f) #history i kaydediyoruz

#%% Load history     #burda ise history i yüklüyoruz
"""import codecs
with codecs.open("mnist_json_agirlik/cnn_mnist.json","r",encoding="utf-8") as f:
    h=json.loads(f.read()) #history i yüklüyoruz

#%% bu kisimda ise h yi yükleyerek tekrar görselleştirme yapabiliriz
plt.plot(h["loss"],label="Train loss")
plt.plot(h["val_loss"],label="Validation loss")
plt.legend()
plt.show()

plt.figure()

plt.plot(h["accuracy"],label="Train acc")
plt.plot(h["val_accuracy"],label="Validation acc")
plt.legend()
plt.show()"""
