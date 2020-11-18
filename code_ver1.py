# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../input/ctcmodel/')

#import file

from keras.models import Sequential
from keras.layers import Input, Activation, TimeDistributed
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.fftpack import dct
import os
from keras.layers import Bidirectional
from CTCModel import CTCModel as CTCModel
import tensorflow as tf
import keras.backend as K


def prepare_output():
  file_path = '../input/gujarai-data-set-a/gujarati_train_text'
  file = open(file_path)
  out_dict={}
  line_num=1
  for l in file:
    audio_num = l.split(' ')[0]
    text_line = l.split(' ')[0:]
    text_line[0] = text_line[0][10:]
    text_stitched = ""
    for word in text_line:
      text_stitched += word+ ' '
    audio_num = audio_num[:9]
    out_dict[audio_num]=text_stitched
    line_num+=1
  return out_dict

def convert_input(sample_rate, signal):
  pre_emphasis = 0.97
  emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
  frame_size = 0.025
  frame_stride = 0.010
  frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
  signal_length = len(emphasized_signal)
  frame_length = int(round(frame_length))
  frame_step = int(round(frame_step))
  num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
  pad_signal_length = num_frames * frame_step + frame_length
  z = np.zeros((pad_signal_length - signal_length))
  pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

  indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
  frames = pad_signal[indices.astype(np.int32, copy=False)]
  frames *= np.hamming(frame_length)
  NFFT = 512
  mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
  pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
  nfilt = 80
  low_freq_mel = 0
  high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
  mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
  hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
  bin = np.floor((NFFT + 1) * hz_points / sample_rate)

  fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
  for m in range(1, nfilt + 1):
      f_m_minus = int(bin[m - 1])   # left
      f_m = int(bin[m])             # center
      f_m_plus = int(bin[m + 1])    # right

      for k in range(f_m_minus, f_m):
          fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
      for k in range(f_m, f_m_plus):
          fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
  filter_banks = np.dot(pow_frames, fbank.T)
  filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
  filter_banks = 20 * np.log10(filter_banks)  # dB
  data = np.array(filter_banks, dtype = float)
  maxx = np.max(data)
  data = np.divide(data,maxx)
  return data

char_dict={}
jstart=1

def line_to_charset(line_dict, unclassified_label):
  extra_dict={}
  # j=91
  global jstart
  global char_dict
  output_arr=[]
  for strings in line_dict:
    string_to_num =[]
    for character in strings:
      if character in char_dict:
        string_to_num.append(char_dict[character])
      else:
        if character not in extra_dict:
          extra_dict[character]=jstart
          string_to_num.append(jstart)
          jstart+=1
        else:
          string_to_num.append(extra_dict[character])
    # print(string_to_num)
    output_arr.append(string_to_num)
  char_dict.update(extra_dict)
  extra_dict={}
  return output_arr


def prepare_input_output(out_dict,start_index,batch_size):
  file_path  = '../input/gujarai-data-set-a/PartA_Gujarati/PartA_Gujarati/Train/Audio'
  file_folder = os.listdir(file_path)
#   print('total num of files is %d' %len(file_folder))
#   print('start index is %d and batch size is %d' %(start_index,batch_size))
  count=0
  x_train=[]
  y_train=[]
  i = start_index
  while i<len(file_folder):
    filez=file_folder[i]
    file_name = filez.split('.')[0]
#     print('i is %d' %i)
    i+=1
    if count>=batch_size:
        break
    if count<batch_size and file_name in out_dict:
#       print('count is %d' %count)    
      sample_rate, signal = scipy.io.wavfile.read(os.path.join(file_path,filez))
      x_data = convert_input(sample_rate, signal)
      x_train.append(x_data)
      y_train.append(out_dict[file_name])
      count+=1
    else:
      continue
#   start_index+=batch_size
  y_train = line_to_charset(y_train,90)
  y_train = np.array(y_train)
  x_train = np.array(x_train)  
  lmax=0
  l2max=0
#   print(x_train.shape)
#   print(y_train.shape)  
#   print(x_train[0].shape)
#   print(len(y_train[0]))
  for i in range(len(x_train)):
    l1 = x_train[i].shape[0]
    l2 = len(y_train[i])
    lmax=max(l1,lmax)
    l2max=max(l2,l2max)
  for i in range(len(x_train)):
    l1,l2 = x_train[i].shape
    l3 = len(y_train[i])
    diff = lmax-l1
    diff2=l2max-l3
    zrs = np.zeros(shape=(diff,l2)).astype(np.float32)
    x_train[i]=np.concatenate([x_train[i],zrs],0)
    y_train[i]=np.asarray(y_train[i]).astype(np.float32)
    zrs2 = np.zeros(diff2).astype(np.float32)
    if diff2>0:
#         print('print type of y_train[i] is %s' %(y_train[i]))
#         print('print shape of y_train[i] is %s' %(y_train[i].shape))
        y_train[i]=np.concatenate([y_train[i],zrs2])
#         print('print type of y_train[i] is %s' %(type(y_train[i])))
#         print('print length of y_train[i] is %s' %(y_train[i].shape))
  for i in range(len(x_train)):
    x_train[i]=x_train[i].reshape(1,x_train[i].shape[0],x_train[i].shape[1])
  x_train_dash = tf.concat([x_train[i] for i in range(len(x_train))], axis=0)
#   print(x_train_dash.dtype)
#   print(x_train_dash.shape)
  for i in range(len(y_train)):
#     if start_index==256 or start_index==254:
#         print(y_train[i].shape)
#         print(type(y_train[i]))
    y_train[i]=np.array(y_train[i]).reshape(1,-1)
#     y_train[i]=y_train[i].reshape(1,-1)
#     y_train.resize(1,len(y_train[i]))
#     if start_index==256 or start_index==254:
#         print(y_train[i].shape)
#         print(type(y_train[i]))
  y_train_dash = tf.concat([y_train[i] for i in range(len(y_train))], axis=0)
#   print(y_train_dash.dtype)
#   print(y_train_dash.shape)
  x_train_len = np.asarray([len(x_train_dash[i]) for i in range(len(x_train_dash))])
  y_train_len = np.asarray([len(y_train_dash[i]) for i in range(len(y_train_dash))])
#   print(x_train_len)
#   print(y_train_len)
  return [x_train_dash,y_train_dash,x_train_len,y_train_len]
      
def create_model():
#     global char_dict
    length_char_dict=93 #because unicode representation of Gujarati has 91 characters
    input_layer = Input(name = 'input',shape=(None,80))
    bilstm0 = Bidirectional(LSTM(180,return_sequences=True))(input_layer)
    bilstm1 = Bidirectional(LSTM(180,return_sequences=True))(bilstm0)
    bilstm2 = Bidirectional(LSTM(100,return_sequences=True))(bilstm1)
#     bilstm3 = Bidirectional(LSTM(240,return_sequences=True))(bilstm2)
    dense = TimeDistributed(Dense(length_char_dict))(bilstm2)
    output_layer = Activation('sigmoid', name='sigmoid')(dense)
    network_model = CTCModel([input_layer],[output_layer])
    network_model.compile(optimizer=Adam(lr=0.5*1e-4))
#     network_model.summary()
    
#     test_function = K.function([net_input], [output])
    return network_model
    
out_dict = prepare_output()
network_model=create_model()
# network_model.load_model('../input/iter-wt-final-2/iter_wt_final_2_1',optimizer=Adam(lr=0.5*1e-3))

# x_train_len = np.asarray([len(x_train_dash[i]) for i in range(len(x_train_dash))])
# y_train_len = np.asarray([len(y_train_dash[i]) for i in range(len(y_train_dash))])
#x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
num_epochs=50
# tensorboard_callbackz = tf.keras.callbacks.TensorBoard(log_dir="./logs_2", update_freq='epoch')
train_loss=[]
val_loss=[]
my_path='./'+'iter_wt_final_2_'+os.path.join(str(1))
if not os.path.exists(my_path):
    os.mkdir(my_path)
b_size=8
file_path  = '../input/gujarai-data-set-a/PartA_Gujarati/PartA_Gujarati/Train/Audio'
file_folder = os.listdir(file_path)
tot_data_len=len(file_folder)
for i in range(num_epochs):
    start_index=0
#     if i%10==0:
#         history = network_model.train_on_batch(x=[x_train_dash,y_train_dash,x_train_len,y_train_len], y=np.zeros(len(x_train_dash)), epochs = 1,
#                          verbose=2, validation_split=0.15, batch_size=b_size)
#     else:
    num_batches_per_epoch = tot_data_len//b_size
    train_loss_per_epoch=[]
    val_loss_per_epoch=[]
    for j in range(num_batches_per_epoch):
        x_train_dash,y_train_dash,x_train_len,y_train_len=prepare_input_output(out_dict,start_index, b_size)
        start_index+=b_size
        history = network_model.train_on_batch(x=[x_train_dash,y_train_dash,x_train_len,y_train_len], y=np.zeros(len(x_train_dash)))
        print('loss on batch is %f' %(history))
#         print(history.metrics_names)
        train_loss_per_epoch.append(history)
#         val_loss_per_epoch.append(history.history['val_loss'])
    if i%3==0:
        network_model.save_model(my_path)
    loss_this_epoch = np.mean(train_loss_per_epoch)
    train_loss.append(loss_this_epoch)
    print('epoch num is %d and average loss this epoch is %f' %(i,loss_this_epoch))
#     val_loss.append(np.sum(val_loss_per_epoch))
        

