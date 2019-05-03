#import image_ocr_osman
import os
import itertools
import codecs
import re
import datetime
import editdistance
import numpy as np
from scipy import ndimage
import pylab
import cairocffi as cairo
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks



weight_file = 'weights24.h5'
def create_model(weight_file):
    img_w = 128
    # Input Parameters
    img_h = 64
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))
    
    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    minibatch_size = 32
    
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
    
    fdir = os.path.dirname(get_file('wordlists.tgz',
                                    origin='http://www.mythic-ai.com/datasets/wordlists.tgz', untar=True))
    
    img_gen = TextImageGenerator(monogram_file='yeni_monoOsman3.txt',#os.path.join(fdir, 'wordlist_mono_clean.txt'),
                                 bigram_file='yeni_biOsman2.txt',#os.path.join(fdir, 'wordlist_bi_clean.txt'),
                                 minibatch_size=minibatch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=(pool_size ** 2),
                                 val_split=words_per_epoch - val_words
                                 )
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
    
    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
    
    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
    
    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    
    # transforms RNN output to character activations:
    inner = Dense(img_gen.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    #Model(inputs=input_data, outputs=y_pred).summary()
    
    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.load_weights(weight_file, by_name=True)
    
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])
    
    """from keras.utils import plot_model
    plot_model(model, to_file='model1.png', show_shapes=True)
    from IPython.display import Image
    Image(filename='model1.png')"""
    
    model_p = Model(inputs=input_data, outputs=y_pred)
    return model_p


def decode_predict_ctc(out, top_paths = 1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
      beam_width = top_paths
    for i in range(top_paths):
      lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
                           greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
      text = labels_to_text(lables)
      text = arabic_reshaper.reshape(text)
      results.append(text)
    return results
  
def predit_a_image(a, top_paths = 1):
  c = np.expand_dims(a.T, axis=0)
  net_out_value = model_p.predict(c)
  top_pred_texts = decode_predict_ctc(net_out_value, top_paths)
  return top_pred_texts


import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import arabic_reshaper

model_p = create_model(weight_file)
def text_to_label_no_alphabet(text):
    ret = []
    for char in text:
        ret.append(char)
   
    return ret

#sondakini sil        
def drop_one(s):
    ta = list(s)
    if (len(ta)-1)<1:
        return ''
    ta.pop(len(ta)-1)
    return ''.join(ta)

#metni ters çevir
def reverse_for_loop(s):
    s = list(s)
    s1 = ''
    for c in s:
        s1 = c + s1  # appending chars in reverse order
    return ''.join(s1)


h = 64
w = 128
text = 'میزان'
#text = arabic_reshaper.reshape(text)
aa = cv2.imread('test.jpg',0)
plt.imshow(aa, cmap='Greys_r')
plt.show()
aa.shape = [1,64,128]
aa =aa.astype('float64')/255
for index, x in np.ndenumerate(aa):
    if aa[index]> 0.95:
        aa[index] = 1
    if aa[index]< 0.1:
        aa[index] = 0
bb = aa.reshape((h, w))
cc = np.expand_dims(aa.T, axis=0)
print('-------')
print(cc.shape)
net_out_value = model_p.predict(cc)
pred_texts = decode_predict_ctc(net_out_value)


a = paint_text(text,h = h, w = w)
b = a.reshape((h, w))
c = np.expand_dims(a.T, axis=0)
net_out_value = model_p.predict(c)
pred_texts = decode_predict_ctc(net_out_value)

tahmin_yazi = predit_a_image(aa, top_paths = 3)
print(tahmin_yazi)

ex1 = pd.read_csv('data/pred/temiz_data_sayi.csv')
ex = pd.read_csv('data/pred/temiz_data.csv')

bulunan = []
#kendisi eşleşiyormu bul
ilk_metin =  ex.loc[ex['osmanlica'].isin(tahmin_yazi)]
"""print(ilk_metin.count())
print('--------------------------------')"""
if ilk_metin['turkce'].count() > 0:
    print(ilk_metin)
#harf eksilt bul
if True:
    #ilk harfi eksilt
    tahminler1 =[drop_one(x) for i,x in enumerate(tahmin_yazi)]
    ters_metin =  ex.loc[ex['osmanlica'].isin(tahminler1)]
    #ilk harfi eksiltilmişi arat
    tahminler1 =[reverse_for_loop(x) for i,x in enumerate(tahminler1)]
    
    eksik_harfli = ex.loc[ex['osmanlica'].isin(tahminler1)]
    #ikinci harfi eksilt
    #print([dropone(x) for i,x in enumerate(tahmin_yazi)])
#ters çevir bul
if True:
    tahminler2 =[reverse_for_loop(x) for i,x in enumerate(tahmin_yazi)]
    ters_metin =  ex.loc[ex['osmanlica'].isin(tahminler2)]
    #ters çevirilmişi eksilt arat
    tahminler2 =[drop_one(x) for i,x in enumerate(tahminler2)]
    ters_metin =  ex.loc[ex['osmanlica'].isin(tahminler2)]
   # tahminler2 =[reverse_for_loop(x) for i,x in enumerate(tahminler2)]
    
    tahminler2 =[arabic_reshaper.reshape(x) for i,x in enumerate(tahminler2)]

    ters_metin =  ex.loc[ex['osmanlica'].isin(tahminler2)]
    
print('Tahminler')
print(pd.concat([ilk_metin,eksik_harfli,ters_metin]))   