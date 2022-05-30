from pathlib import Path
import pandas as pd
import os.path
import numpy as np
from sklearn.model_selection import train_test_split

"""
Input debe ser matriz numerica de 333x333 en vez de imagen de 660*991
"""


#imgpath = 'C:/Users/david/Desktop/Python_test/TFMPaper/Colabs/Fractal_NoFractal'
imgpath = 'C:/Users/david/Desktop/Cuencas/Duffing'

#Path de imagenes a trabajar siendo su nombre la FDim
def imagedataframegenerator(imgpath):
# Path de imagenes 
    image_dir = Path(imgpath)
#Genera el dataframe images, compuesta de los paths de las imagenes y del nombre de cada imagen (la f dim)
    filepaths = pd.Series(list(image_dir.glob(r'**/*')), name='Filepath').astype(str)
    fdim = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[1])[1].replace('.png','')), name= 'Dimension_Fractal').astype(np.double)
    images = pd.concat([filepaths, fdim], axis=1).sample(frac=1.0, random_state=1) #el .sample(...) hace el shuffle
#We can use less imagenes if needed
    image_df = images.sample(len(images), random_state=1).reset_index(drop=True)
    return image_df

image_df = imagedataframegenerator(imgpath)
train_df, test_df = train_test_split(image_df, train_size=0.8, shuffle=True, random_state=0) #Coge de todos los datos el 80% para entrenar

print('Total de imagenes',len(image_df))

print('Imagenes de entrenamiento',len(train_df))
print('Imagenes de test',len(test_df))

print("\n")

import tensorflow as tf

#Data augmentation techniques

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, #Del train me coge un 20% para validar, de forma que ya hay train, val, y test
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

image_size_x = 64 #660 #MATLAB GUARDA LAS IMAGENES CON ESTAS DIMENSIONES EN PIXELES (PHOTOSHOP)
image_size_y = 64 #891

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Dimension_Fractal',
    target_size=(image_size_x,image_size_y),
    crop_to_aspect_ratio=True,
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Dimension_Fractal',
    target_size=(image_size_x,image_size_y),
    crop_to_aspect_ratio=True,
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Dimension_Fractal',
    target_size=(image_size_x,image_size_y),
    crop_to_aspect_ratio=True,
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)

print("\n")

"""
#from matplotlib import pyplot as plt
#plt.imshow(np.squeeze(train_images[0][0]), interpolation='nearest')
#plt.show()
"""

# Aqui se define toda la arquitectura de la ResNet, en la carpeta del codigo hay una imagen con un esquema de la arquitectura de la ResNet 


from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform

def identity_block(X, f, filters, stage, block):

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Saving the input value.we need this later to add to the output. 
    X_Residual = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X, X_Residual])
    X = Activation('relu')(X)
    
    
    return X
def convolutional_block(X, f, filters, stage, block, s = 2):
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_Residual = X


    # First layer 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a')(X) # 1,1 is filter size
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)  # normalization on channels
    X = Activation('relu')(X)

      
    # Second layer  (f,f)=3*3 filter by default
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    # Third layer
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


    # Residual Connection
    X_Residual = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1')(X_Residual)
    X_Residual = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_Residual)

    # Final step: Add shortcut value here, and pass it through a RELU activation 
    X = Add()([X, X_Residual])
    X = Activation('relu')(X)
    
    
    return X
#Each ResNet block is either 2 layer deep
def ResNet50(input_shape=(64, 64, 3), outputs=1):

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input) #3,3 padding

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X) #64 filters of kernel size 7*7 pixels
    X = BatchNormalization(axis=3, name='bn_conv1')(X) #batchnorm applied on channels
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X) #window size is 3*3

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b') 
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL 
    X = AveragePooling2D((2,2), name="avg_pool")(X)

    # output layer
    X = Flatten()(X)
    X = Dense(outputs, activation='linear', name='fc' + str(outputs), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

model = ResNet50(input_shape = (64, 64, 3), outputs = 1)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
optimizer = tf.keras.optimizers.Adam()
lr_metric = get_lr_metric(optimizer)

model.compile(optimizer=optimizer, loss='mse', metrics=[lr_metric])

model.summary()


#Guarda como checkpoint los pesos de la red neuronal en cada iteraciones https://www.tensorflow.org/tutorials/keras/save_and_load
checkpoint_path = "C:/Users/david/Desktop/Python_test/TFMPaper/Python_Runners/FDimResNet/CheckpointPath/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
callbacks_list = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

"""
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=3,
    callbacks= callbacks_list,
)


model.save('C:/Users/david/Desktop/CustomCNN/results/LinearModel.h5')
np.save('C:/Users/david/Desktop/CustomCNN/results/history1.npy',history.history)
history1=np.load('C:/Users/david/Desktop/CustomCNN/results/history1.npy',allow_pickle='TRUE').item()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib
from matplotlib import pyplot as plt

test_predictions = model.predict(train_images).flatten()

plt.scatter(train_images.labels,test_predictions)
plt.xlabel('True Fractal Dimension')
plt.ylabel('Predicted Fractal Dimension')
plt.axis('equal')
plt.axis('square')
_ = plt.plot([1,2], [1,2])
plt.savefig('C:/Users/david/Desktop/CustomCNN/results/PredictionRegressorFunction.PNG') #SAVEFIG VA ANTES DE SHOW
plt.savefig('C:/Users/david/Desktop/CustomCNN/results/PredictionRegressorFunction.eps') #SAVEFIG VA ANTES DE SHOW
plt.show()

error = test_predictions - train_images.labels
plt.hist(error, bins = 30)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.savefig('C:/Users/david/Desktop/CustomCNN/results/PredictioHist.PNG') #SAVEFIG VA ANTES DE SHOW
plt.savefig('C:/Users/david/Desktop/CustomCNN/results/PredictioHist.eps') #SAVEFIG VA ANTES DE SHOW
plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------------------
#RECORDAR CAMBIAR TRAIN POR TEST AQUI Y DESPUES
from sklearn.metrics import r2_score

predicted_df = np.squeeze(model.predict(test_images))
true_df = test_images.labels
 
rmse = np.sqrt(model.evaluate(test_images,verbose=1))
#print(rmse)
print("Test RMSE: {:.5f}".format(rmse[0])) #Desviacion tipica?
 
r2 = r2_score(true_df, predicted_df)
print("Test R^2 Score: {:.5f}".format(r2)) 

#-----------------------------------------------------------------------------------------------------------------------------------------------------------


# plot the loss
scale_factor = 1
plt.plot(history1['loss'], color='blue',label='train loss')
plt.plot(history1['val_loss'], color='red', label='val loss')
plt.legend(fontsize=15)

# Set tick font size
plt.xticks(fontsize=15, rotation=0)
plt.yticks(fontsize=15, rotation=90)

xmin, xmax = plt. xlim()
ymin, ymax = plt. ylim()
plt.title("Evolution of the loss function", fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)

plt. xlim(xmin * scale_factor, xmax * scale_factor)
plt. ylim(ymin * scale_factor, ymax * scale_factor)

plt.savefig('C:/Users/david/Desktop/CustomCNN/results/LossFunction.png') #SAVEFIG VA ANTES DE SHOW
plt.savefig('C:/Users/david/Desktop/CustomCNN/results/LossFunction.eps') #SAVEFIG VA ANTES DE SHOW
plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------------------
"""