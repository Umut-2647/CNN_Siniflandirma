from keras.models import Sequential #type:ignore
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense #type:ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img #type:ignore
import matplotlib.pyplot as plt 
from glob import glob
import warnings
warnings.filterwarnings("ignore")


train_path= "archive/fruits-360_100x100/fruits-360/Training/" #train yolunu yaziyourz
test_path = "archive/fruits-360_100x100/fruits-360/Test/" #tset yolunu yaziyoruz

img= load_img(train_path+"Apple Braeburn 1/0_100.jpg") #bir resim deniyoruz
"""plt.imshow(img)
plt.axis("off")
plt.show()"""

x= img_to_array(img) #resmi arraya çeviriyoruz
print(x.shape)


className= glob(train_path+"/*") #train yolundaki klasörleri aliyoruz
numberOfClass= len(className) #kaç tane klasör olduğunu buluyoruz yani kac tane class olduğunu buluyoruz
print("Number of class : ",numberOfClass)


#%% CNN Model
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=x.shape)) #kerasta yazarken muhakkak shape veriyoruz
model.add(Activation("relu")) #aktivasyon fonksiyonunu ekliyoruz
model.add(MaxPooling2D()) #maxpooling ekliyoruz eger deger vermezsek 2x2 lik bir pooling ekliyoruz


model.add(Conv2D(32,(3,3))) #ikinci conv layer ekliyoruz
model.add(Activation("relu"))
model.add(MaxPooling2D())


model.add(Conv2D(64,(3,3))) #ikinci conv layer ekliyoruz
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024)) #1024 nöronlu bir layer ekliyoruz
model.add(Activation("relu")) 
model.add(Dropout(0.5)) #overfitting i önlemek için dropout ekliyoruz (1024 den 512 ye düşürüyoruz)
model.add(Dense(numberOfClass)) #output layer ekliyoruz
model.add(Activation("softmax"))


model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"]) #modeli compile ediyoruz

batch_size=32 #batch size belirliyoruz


#%% Datya generattion -TRAIN-TEST

train_datagen=ImageDataGenerator(rescale=1./255,
        shear_range=0.3,horizontal_flip=True,zoom_range=0.3)

#resccale= resimleri 0-1 arasina çekiyoruz
#shear_range= resimleri sağ ya da sola kaydiriyoruz
#horizontal_flip= resimleri yatayda çeviriyoruz

test_datagen=ImageDataGenerator(rescale=1./255) #rescale= resimleri 0-1 arasina çekiyoruz cunku train verisi ile ayni olmasi lazim yoksa sonuclar kottu oplur

train_generator=train_datagen.flow_from_directory(train_path,target_size=x.shape[:2],batch_size=batch_size
                                        ,color_mode="rgb",class_mode="categorical")
print("Gerçek sinif sayisi:", train_generator.num_classes)

test_generator=test_datagen.flow_from_directory(test_path,target_size=x.shape[:2],batch_size=batch_size
                                        ,color_mode="rgb",class_mode="categorical")

#flow_directory= resimleri kendisi otomatik olarak eger klasörlere ayrildiysa okuyor
#target_size= resimlerin boyutunu belirliyoruz
#color_mode= resimlerin renk türünü belirliyoruz
#class_mode= resimlerin birden fazla class oldugu için categorical yapiyoruz

hist = model.fit(train_generator,
                    steps_per_epoch = 1600//batch_size,
                    epochs=110,
                    validation_data=test_generator,validation_steps=800//batch_size)


#%% Model Save
model.save("cnn_fruit360.h5") #modeli kaydediyoruz

#%% Model Evaluation
print(hist.history.keys()) #hist parametresi neleri tutuyor onu gösteriyoruz
#hist parametresi sunları tutyor: loss, accuracy, val_loss, val_accuracy
plt.plot(hist.history["loss"],label="Train loss")
plt.plot(hist.history["val_loss"],label="Validation loss")
plt.legend()
plt.show()

plt.figure()

plt.plot(hist.history["accuracy"],label="Train acc")
plt.plot(hist.history["val_accuracy"],label="Validation acc")
plt.legend()
plt.show()

#%% Save history     ##burda kaydetme onemli cuz her seferinde modeli tekrar çalıştırmak yerine sadece history i yükleyerek görebiliriz
import json
with open("cnn_fruit360.json","w") as f:
    json.dump(hist.history,f) #history i kaydediyoruz

#%% Load history     #burda ise history i yüklüyoruz
import codecs
with codecs.open("cnn_fruit360.json","r",encoding="utf-8") as f:
    h=json.loads(f.read()) #history i yüklüyoruz

#%%
plt.plot(h["loss"],label="Train loss")
plt.plot(h["val_loss"],label="Validation loss")
plt.legend()
plt.show()

plt.figure()

plt.plot(h["accuracy"],label="Train acc")
plt.plot(h["val_accuracy"],label="Validation acc")
plt.legend()
plt.show()
# %%
