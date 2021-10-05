from sys import path
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from glob import glob
import argparse
import warnings
import os
import random
import numpy as np
def split(path_file):
    wav_paths = glob('{}/**'.format(path_file), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    wanted_class=['yes','no','up','down','left','right','on','off','stop','go']
    dt={'wanted':[],'unknown':[],'silence':[]}
    for x in wav_paths:
        lbl=os.path.split(x)[0].split('/')[-1]
        if lbl == 'silence':
            dt['silence'].append([x,'silence'])
        elif lbl not in wanted_class:
            dt['unknown'].append([x,'unknown'])
        else:
            dt['wanted'].append([x,lbl])

    silent_file=dt['silence'][0][0]
    random.shuffle(dt['unknown'])
    unknown_size=len(dt['unknown'])*30//100
    silence_size=len(dt['unknown'])*20//100
    for i in range(silence_size):
        dt['silence'].append([silent_file,'silence'])

    unknown_val=dt['unknown'][:unknown_size]
    silence_val=dt['silence'][:silence_size]
    wanted_val=dt['wanted']
    mydataset=wanted_val+unknown_val+silence_val
    mydataset=np.array(mydataset)
    X=mydataset[:,0]
    y=mydataset[:,1]
    classes = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']
    le = LabelEncoder()
    le.fit(y)
    cl=le.classes_
    labels = le.transform(y)
    silence_index=np.where(cl=='silence')
    wav_train, wav_val, label_train, label_val = train_test_split(X,
                                                                 labels,
                                                                test_size=0.2,
                                                                shuffle=True,
                                                                random_state=245)
    wav_val,wav_test,label_val,label_test=train_test_split(wav_val,label_val,test_size=0.5,shuffle=True,random_state=45)

    return wav_train,wav_test,wav_val,label_train,label_test,label_val,cl

# path='f:\\datas\\kws_streaming\\data2\\'
# s=split(path)