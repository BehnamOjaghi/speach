from numpy.lib.function_base import select
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from scipy.io import wavfile

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=False,silence_inedx=-1,dt_type='mfcc'):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dt_type=dt_type
        # self.silence_inedx=silence_inedx
        self.on_epoch_end()


    def __len__(self):
        # print(self.shuffle)
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):
        # print(index*self.batch_size,'----',(index+1)*self.batch_size)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        if self.dt_type=='mel':
            X = np.empty((self.batch_size,1,int(self.sr*self.dt)), dtype=np.float32) # mel
        else:
            X = np.empty((self.batch_size,int(self.sr*self.dt)), dtype=np.float32) # mfcc
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            # ///// add noise and nomalize ////
            wav = wav.astype(np.float)
            # if self.silence_inedx==label:
            #     wav*=0
            # wav/=32768.0                            #normalize data [-1..1)
            # wav -= wav.mean()
            # val= np.max((wav.max(), -wav.min()))
            # if val !=0:
            #     wav /=val
            # add gaussian noise
            wav += np.random.normal(loc=0.0, scale=0.025, size=wav.shape)
            # /////////////////////////////////
            if wav.shape[0]<self.sr:      #smaller
                randPos = np.random.randint(self.sr-wav.shape[0])
                if self.dt_type=='mel':
                    X[i,0,randPos:randPos + wav.shape[0]]=wav
                else:
                    X[i,randPos:randPos + wav.shape[0]]=wav

            else:
                X[i,] = wav
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)