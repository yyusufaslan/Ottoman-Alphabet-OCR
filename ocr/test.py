

import matplotlib.pyplot as plt
h = 64
w = 128
import arabic_reshaper
text = 'ا'

#text = arabic_reshaper.reshape(text)
import cv2 
aa = cv2.imread('test3.png',0)
aa.shape = [1,64,128]
aa =aa.astype('float64')/255    
for index, x in np.ndenumerate(aa):
    if aa[index]> 0.95:
        aa[index] = 1
bb = aa.reshape((h, w))
plt.imshow(bb, cmap='Greys_r')
plt.show()
cc = np.expand_dims(aa.T, axis=0)
net_out_value = model_p.predict(cc)
pred_texts = decode_predict_ctc(net_out_value)



a = paint_text(text,h = h, w = w)
b = a.reshape((h, w))
plt.imshow(b, cmap='Greys_r')
plt.show()
c = np.expand_dims(a.T, axis=0)
net_out_value = model_p.predict(c)
pred_texts = decode_predict_ctc(net_out_value)
#pred_texts


#plt.imshow(net_out_value[0].T, cmap='binary', interpolation='nearest')
#plt.show()

tahmin_resim = predit_a_image(aa, top_paths = 3)
tahmin_yazi = predit_a_image(a, top_paths = 3)
print(tahmin_resim)
print(tahmin_yazi)



import pandas as pd
ex1 = pd.read_csv('data/pred/temiz_data_sayi.csv')
ex = pd.read_csv('data/pred/temiz_data.csv')



tahmin_yazi = ['اباب', 'ac', 'ﺍﺁﺍ']
bulunan = []
#kendisi eşleşiyormu bul
ilk_metin =  ex.loc[ex['osmanlica'].isin(tahmin_yazi)]
"""print(ilk_metin.count())
print('--------------------------------')"""
if ilk_metin['turkce'].count() > 0:
    print('--------------------------------')
    #print(ilk_metin)
    print('1 var : ')
    
#harf eksilt bul
if True:
    #ilk harfi eksilt
    print('--------------------------------')
    tahminler =[drop_one(x) for i,x in enumerate(tahmin_yazi)]
    #ilk harfi eksiltilmişi arat
    tahminler =[reverse_for_loop(x) for i,x in enumerate(tahminler)]
    
    eksik_harfli = ex.loc[ex['osmanlica'].isin(tahminler)]
    #ikinci harfi eksilt
    print(tahminler)
    print('2 var : ')
        #print([drop_one(x) for i,x in enumerate(tahmin_yazi)])
#ters çevir bul
if True:
    print('--------------------------------')
    tahmin_yazi =[reverse_for_loop(x) for i,x in enumerate(tahmin_yazi)]
    #ters çevirilmişi eksilt arat
    tahmin_yazi =[drop_one(x) for i,x in enumerate(tahmin_yazi)]
    ters_metin =  ex.loc[ex['osmanlica'].isin(tahmin_yazi)]
    print('3 var : ')
    
print(pd.concat([ilk_metin,eksik_harfli,ters_metin]))    


#harf harf ara bul 








        
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