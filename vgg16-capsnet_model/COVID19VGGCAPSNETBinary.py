# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 00:13:39 2020

@author: shamik.tiwari
"""

from __future__ import print_function
from keras import backend as K
from keras import activations
from keras import utils
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras import regularizers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

IMG_SIZE = 299

def DataGenerator(train_batch, val_batch, IMG_SIZE):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                 rescale=1.0/255.0,
                                 rotation_range=10,
                                 horizontal_flip=True,
                                 vertical_flip=False)

    datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)

    train_gen = datagen.flow_from_directory('./cov/train',
                                            target_size=(IMG_SIZE, IMG_SIZE),
                                            color_mode='rgb', 
                                            class_mode='categorical',
                                            batch_size=train_batch)

    val_gen = datagen.flow_from_directory('./cov/val', 
                                          target_size=(IMG_SIZE, IMG_SIZE),
                                          color_mode='rgb', 
                                          class_mode='categorical',
                                          batch_size=val_batch)

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                 rescale=1.0/255.0)
    
    datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)

    test_gen = datagen.flow_from_directory('./cov/test', 
                                           target_size=(IMG_SIZE, IMG_SIZE),
                                           color_mode='rgb', 
                                           batch_size=1,
                                           class_mode='categorical',
                                           shuffle=False)
    
    return train_gen, val_gen, test_gen

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1 
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
    
class Capsule(Layer):
    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3, 
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
    

from keras.layers import Input
input_image = Input(shape=(299,299, 3))

base_model = VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=(299,299,3)))

base_model.summary()
for layer in base_model.layers:
    layer.trainable = False
    print(layer, layer.trainable)

output = Conv2D(256, kernel_size=(9,9), strides=(1, 1), activation='relu')(base_model.output)

x = Reshape((-1, 256))(output)
capsule = Capsule(2, 16, 4, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
VGGCapsnet_Binarymodel = Model(base_model.input, outputs=output)

VGGCapsnet_Binarymodel.summary()

train_batch = 32
val_batch = 2
train, val, test = DataGenerator(train_batch, val_batch, IMG_SIZE)
lr=1e-4
# lr=1e-3

checkpoint = ModelCheckpoint("weights.h5", 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=False, 
                             mode='min')

# early = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)
early = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

callback_list = [checkpoint, early]

epochs=100

# VGGCapsnet_Binarymodel.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
VGGCapsnet_Binarymodel.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr,decay=0.001, momentum=0.9), metrics=['accuracy'])
# VGGCapsnet_Binarymodel.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr), metrics=['accuracy'])

hist=VGGCapsnet_Binarymodel.fit_generator(train,
                    epochs=epochs,
                    validation_data=val, 
                    validation_steps = len(val.classes)//val_batch,
                    steps_per_epoch=len(train.classes)//train_batch,
                    callbacks=callback_list)
    

accuracy = hist.history['acc']
val_accuracy = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'go-', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'ms-', label='Validation accuracy')
plt.title('Training and validation accuracy curves for VGGCapsNet Model')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.figure()
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'rs-', label='Validation loss')
plt.title('Training and validation loss curves for VGGCapsNet Model')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

X_test, y_test = next(test)

pred=VGGCapsnet_Binarymodel.predict(X_test,verbose=1)
predicted_classes = np.argmax(np.round(pred),axis=1)
tar = np.argmax(np.round(y_test),axis=1)
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(2)]
print(classification_report(tar, predicted_classes, labels=None, target_names=None))

from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle
n_classes=2
lw=2
y_score = pred
target=y_test

### MACRO
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(target[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=3)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
classes = ['Covid19', 'Non-Covid']
for (i, j, color) in zip(range(n_classes),classes, colors):
       plt.plot(fpr[i], tpr[i], color=color, lw=lw,label=j+'(area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

    
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for Covid-19 detection using VGGCapsNet model')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.style
import seaborn as sns;
import matplotlib as mpl

# encoder = LabelEncoder()
# encoder.fit(classes)

mpl.style.use('seaborn-paper')
x=confusion_matrix(y_test.argmax(axis=1),pred.argmax(axis=1))
# Cm_df=pd.DataFrame(x,index=encoder.classes_,columns=encoder.classes_)
Cm_df=pd.DataFrame(x)
sns.set(font_scale=1.5,color_codes=True,palette='deep')
sns.heatmap(Cm_df,annot=True,annot_kws={'size':16},fmt='d',cmap='YlGnBu')
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title('Confusion Matrix')
# plt.show()

