from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D
    
from keras.utils.data_utils import get_file
from keras.models import Model
from keras import layers
from keras.preprocessing import image
import numpy as np
from scipy.spatial.distance import cosine
from  torch.autograd import Variable
import yaml
import os
import cv2

def Unit_Block(tensor, kernel_shape, layers,
                          bias=False):
    layer1, layer2, layer3 = layers

    x = Conv2D(layer1, (1, 1), use_bias=bias,)(
        tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(layer2, kernel_shape, use_bias=bias,
               padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(layer3, (1, 1), use_bias=bias)(x)
    x = BatchNormalization(axis=3)(x)

    # concat outputs here
    x = layers.add([x, tensor])
    x = Activation('relu')(x)
    return x


def Resnet_Block(tensor, kernel_shape, layers,
                      stride=(2, 2), bias=False):
    layer1,layer2,layer3=layers
    
    x = Conv2D(layer1, (1, 1), strides=stride, use_bias=bias
               )(tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(layer2, kernel_shape, padding='same', use_bias=bias,
               )(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(layer3, (1, 1),  use_bias=bias)(x)
    x = BatchNormalization(axis=3)(x)

    skip_connection = Conv2D(layer3, (1, 1), strides=stride, use_bias=bias,
                      )(tensor)
    skip_connection = BatchNormalization(axis=3)(
        skip_connection)

    x = layers.add([x, skip_connection])
    x = Activation('relu')(x)
    return x


def RESNET50():
    input_shape = (224,224,3)

    foto_input =Input(tensor=None, shape=input_shape)

    print("input shape : ",input_shape)
    x = Conv2D(
        64, (7,7), use_bias=False, strides=(2, 2), padding='same'
      )(foto_input)

    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    print("First block output shape: ",x.shape)


    x = Resnet_Block(x, 3, [64, 64, 256],stride=(1, 1))
    print("Second block output shape: ",x.shape)
    x = Unit_Block(x, 3, [64, 64, 256])
    print("Third block output shape: ",x.shape)
    x = Unit_Block(x, 3, [64, 64, 256])
    print("Fourth block output shape: ",x.shape)
    x = Resnet_Block(x, 3, [128, 128, 512])
    print("Fifth block output shape: ",x.shape)
    x = Unit_Block(x, 3, [128, 128, 512])
    print("Sixth block output shape: ",x.shape)
    x = Unit_Block(x, 3, [128, 128, 512])
    print("Seventh block output shape: ",x.shape)
    x = Unit_Block(x, 3, [128, 128, 512])
    print("Eight block output shape: ",x.shape)
####
    x = Resnet_Block(x, 3, [256, 256, 1024])
    print("Ninth block output shape: ",x.shape)

    x = Unit_Block(x, 3, [256, 256, 1024])
    print("Tenth block output shape: ",x.shape)
    x = Unit_Block(x, 3, [256, 256, 1024])
    print("Eleventh block output shape: ",x.shape)
    x = Unit_Block(x, 3, [256, 256, 1024])
    print("Twelveth block output shape: ",x.shape)
    x = Unit_Block(x, 3, [256, 256, 1024])
    print("Thirteenth block output shape: ",x.shape)
    x = Unit_Block(x, 3, [256, 256, 1024])
    print("Fourteenth block output shape: ",x.shape)
####
    x = Resnet_Block(x, 3, [512, 512, 2048])
    print("Fifteenth block output shape: ",x.shape)
    x = Unit_Block(x, 3, [512, 512, 2048])
    print("Sixteenth block output shape: ",x.shape)
    x = Unit_Block(x, 3, [512, 512, 2048])
    print("Seventeenth block output shape: ",x.shape)
    x = AveragePooling2D((7, 7))(x)
    x = GlobalMaxPooling2D()(x)
    print("Last block output: ",x.shape)

    model = Model(foto_input, x, name='vggface_resnet50')
    weights_path = get_file('resnet_parametreler.h5',
                                    'https://github.com/kemaleddin222/resnet_param/raw/master/resnet_parametreler.h5',
                                    cache_subdir='./')
    model.load_weights(weights_path)
    return model



def start(policedb,criminal_path):
    model=RESNET50()
    gan_res=image.load_img(criminal_path,(224,224))

    x = image.img_to_array(gen_res)
    pyplot.imshow(x/255)
    pyplot.show()

    x = np.expand_dims(x, axis=0)
    criminal=model.predict(x) #Criminal vector is here
	
    dirs=os.listdir(policedb)
    final_scores={}
    
    for dir in dirs:
        imgs=os.listdir(policedb+dir)
        scores=[]
        for img in imgs:
            img = image.load_img(folder+dir+"/"+img)
            x = image.img_to_array(img)
            x=cv2.resize(x,(224,224))
            x = np.expand_dims(x, axis=0)
            res1=model.predict(x)
            scores.append(1-cosine(res1[0],criminal[0]))
        final_scores[max(scores)]=dir
    
    maxs=[x for x in sorted(final_scores.keys(),reverse=True)]
    final=[]
    for x in range(3):
        imgs=os.listdir(policedb+final_scores[maxs[x]])
        final.append(pyplot.imread(policedb+final_scores[maxs[x]]+"/"+imgs[0]))

    print("Most similar results :")
    pyplot.subplot(131)
    pyplot.imshow(final[0])
    print("Similarity score  : {:.3f} name : {}".format(maxs[0]*100,final_scores[maxs[0]]))
    pyplot.subplot(132)
    pyplot.imshow(final[1])
    print("Similarity score  : {:.3f} name : {}".format(maxs[1]*100,final_scores[maxs[1]]))
    pyplot.subplot(133)
    pyplot.imshow(final[2])
    print("Similarity score  : {:.3f} name : {}".format(maxs[2]*100,final_scores[maxs[2]]))
    pyplot.show()


