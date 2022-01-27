import numpy as np
# import tensorflow as tf
from keras import backend as K
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import *
from keras.models import Model
import PIL


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
                 num_capsule=2,
                 dim_capsule=16,
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


def predict_input(image):
    K.clear_session()

    train_batch = 32
    val_batch = 2
    # train, val, test = DataGenerator(train_batch, val_batch, IMG_SIZE)

    reconstructed_model = keras.models.load_model("weights.h5", custom_objects={'Capsule': Capsule})

  
    # image = PIL.Image.open("./cov/test/NON-COVID/41.jpeg")
    image = PIL.Image.fromarray(image)
    new_image = image.resize((299, 299))
    image_array = np.array(new_image, dtype=np.float32)
    if len(image_array.shape) < 3:
        new_img_arr = np.zeros((299, 299, 3), dtype=np.float32)
        new_img_arr[:,:,0] = image_array
        new_img_arr[:,:,1] = image_array
        new_img_arr[:,:,2] = image_array
        image_array = new_img_arr
    image_array = np.expand_dims(image_array, axis=0)
    print(image_array.shape)

    pred=reconstructed_model.predict(image_array,verbose=1)
    predicted_classes = np.argmax(np.round(pred),axis=1)
    print(pred)
    print(predicted_classes)
    return pred, predicted_classes

if __name__ == "__main__":
    image = PIL.Image.open("./cov/test/NON-COVID/42.jpeg")
    predict_input(image)
    pass