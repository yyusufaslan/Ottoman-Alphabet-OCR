import pandas as pd
ex = pd.read_excel('ocr/data/osmanlica.xlsx')

"""for index, row in ex.iterrows():
    print (row["Turkce"], row["Osmanlica"])
    print('--')"""

#TEMİZLEME BAŞLANGIÇ
import codecs
import re    
alphabet = ' ءﺀآأابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهﯓئةی يكۏىۊۀ‬ٌ ّّّ‬ًّ ‬َ ‬ِ ‬ُ ‬ْ ً‬ ًّ‬۰ ٰٔ '
osmanlicalar = []
turkceler = []

temiz_turkceler= []
temiz_osmanlicalar= []

osmanlica_labels=[]
turkce_labels = []

not_valid = []

yeni_osman = []
yeni_turkce = []
son_osman = []
for index, row in ex.iterrows():
    turkce = row["Turkce"]
    turkce = str(turkce).strip()
    temiz_turkceler.append(turkce)
    turkceler.append(turkce)
    
    osmanlica = row["Osmanlica"]
    osmanlicalar.append(osmanlica)
    
    osmanlica = str(osmanlica).strip()
    
    osmanlicalabel = text_to_labels(osmanlica)
    osmanlica_labels.append(osmanlicalabel) 
    for index in osmanlicalabel:
        if 0 == index:
            osmanlicalabel.remove(0)
        if -1 == index:
            osmanlicalabel.remove(-1)
    yeni_osman.append(osmanlicalabel)
    temiz_osmanlicalar.append(labels_to_text(osmanlicalabel))
    
    son_osman.append(text_to_label_no_alphabet(osmanlicalabel))
    if is_valid_str(osmanlica) != True:
        not_valid.append(is_valid_str(osmanlica))
#TEMİZLEME BİTİŞ

#TEMİZ KAYITBAŞLANGIÇ  
import numpy as np
from pandas import DataFrame
"""data = {
        'osmanlica': np.array(son_osman),
        'turkce': temiz_turkceler
        }"""
df = DataFrame(np.array(son_osman))
df2 = DataFrame(temiz_turkceler)
result = pd.concat([df, df2], axis=1, sort=False)
pd.DataFrame(result).to_csv("temiz_data_sayi_kelime.csv")
#TEMİZ KAYIT BİTİŞ


#EK FONKSİYONLAR BAŞLANGIÇ    
def is_valid_str(in_str):
    regex = r'^[\u0600-\u06FF\uFB8A\u067E\u0686\u06AF\u200C\u200F\uFBD3\uFBB2\uFE70\uFE80 ]+$'
    search = re.compile(regex, re.UNICODE).search   
    return bool(search(in_str))


def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret

record = 0
ad = []
def text_to_label_no_alphabet(text):
    ret = []
    for char in text:
        ret.append(char)
    for index in range(32):
        if len(ret) != 32:
            ret.append('0')
        else:
            break
    return ret

def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)
#EK FONKSİYONLAR BİTİŞ
    
#mACHİNE LEARNİNG MODEL

import pandas as pd
ex1 = pd.read_csv('temiz_data_sayi.csv')
ex = pd.read_csv('temiz_data.csv')
osm= ex['osmanlica']
predic = ['انئدود','اشندود','اشنئدود']
search = 'اشنئدود'
#print(ex.loc['Üşni    iDûd','turkce'])
liste=[]

get = [i for i,x in enumerate(osm) if x==tahmin_yazi[0]]
print(ex.loc[get[0],'turkce'])
if(get):
    liste = ex[ex['osmanlica']==search].index.item('turkce')
    print(liste['turkce']==0)
















