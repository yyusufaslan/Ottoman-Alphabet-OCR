

"""weight_file = 'weights24.h5'
def create_model(weight_file):
    img_w = 128
    img_h = 64
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))
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
    inner = Conv2D(conv_filters, kernel_size, padding='same',activation=act, kernel_initializer='he_normal',name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',activation=act, kernel_initializer='he_normal',name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    inner = Dense(img_gen.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.load_weights(weight_file, by_name=True)
    test_func = K.function([input_data], [y_pred])
    model_p = Model(inputs=input_data, outputs=y_pred)
    return model_p"""

import os
from flask import Flask, render_template, request, redirect, abort, flash, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import base64
YUKLEME_KLASORU = 'static/yuklemeler/'
UZANTILAR = set(['png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = YUKLEME_KLASORU
app.secret_key = "Flask_Dosya_Yukleme_Ornegi"


###################MODEL YÜKELEME#########################

###################MODEL YÜKELEME#########################
import image_ocr_osman
import tensorflow as tf
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
from keras.models import load_model
import pandas as pd
import arabic_reshaper

ex1 = pd.read_csv('data/temiz_data_sayi.csv')
ex = pd.read_csv('data/temiz_data.csv')
#############################################################
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
#############################################################
def turkce_getir(tahmin_yazi):

    #kendisi eşleşiyormu bul
    ilk_metin =  ex.loc[ex['osmanlica'].isin(tahmin_yazi)]
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



#############################################################
def decode_predict_ctc(out, top_paths = 3):
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

def predbeyb(yol):
	print(yol)
	aa = cv2.imread(yol,0)
	aa = cv2.resize(aa,(128,64))
	aa.shape = [1,64,128]
	aa =aa.astype('float64')/255
	for index, x in np.ndenumerate(aa):
		if aa[index]> 0.95:
			aa[index] = 1
		if aa[index]< 0.1:
			aa[index] = 0
	cc = np.expand_dims(aa.T, axis=0)
	net_out_value = model_p.predict(cc)
	pred_texts = decode_predict_ctc(net_out_value)
	print(pred_texts)
	return turkce_getir(pred_texts)

model_p = load_model('sdas.h5')


# Ana Sayfa
@app.route("/")
def anaSayfa():
	return "Merhaba API ÇALIŞŞIYOR"

@app.route('/dosyayukle', methods=['POST'])
def dosyayukle():
	data = requests.json
	print(data)

	if request.method == 'POST':	
		data = request.json
		dosya = base64.b64decode(data['dosya'])
		run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
		yol = os.path.join('rrr')
		fh = open(yol+".jpg", "wb")
		fh.write(dosya)
		fh.close()
		top_pred_texts = predbeyb(yol+".jpg")
		return top_pred_texts


#if __name__ == "__main__":
	#model_p = load_model('sdas.h5')
	#model_p = create_model(weight_file)
	#app.run()