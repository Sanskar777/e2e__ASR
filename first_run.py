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
from google.colab import drive
drive.mount('/content/drive')



char_dict={'અ':1,
           'ઑ':2,
           'આ':3,
           'ઇ':5,
           'ઈ':6,
           'ઉ':7,
           'ઊ':9,
           'એ':12,
           'ઐ':14,
           'ઓ':17,
           'ઍ':19,
           'ઔ':21,
           'ક':22,
           'ખ':23,
           'ગ':24,
           'ઘ':25,
           'ઙ':26,
           'ચ':27,
           'છ':28,
           'જ':30,
           'ઝ':31,
           'ઞ':33,
           'ટ':34,
           'ઠ':35,
           'ડ':36,
           'ઢ':37,
           'ણ':38,
           'ત':39,
           'થ':40,
           'દ':41,
           'ધ':42,
           'ન':43,
           'પ':45,
           'ફ':46,
           'બ':47,
           'ભ':48,
           'મ':49,
           'ય':50,
           'ર':51,
           'લ':52,
           'ળ':53,
           'વ':54,
           'શ':55,
           'ષ':56,
           'સ':57,
           'હ':58,
           'ंं':80,
           'ंः':81,
           'ंँ':82
           }

def prepare_output():
  file_path = '/content/gujarati_train_text.txt'
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
    # print('%s with .wav extension' %audio_num)
    # audio_num = os.path.splitext(audio_num)[0]
    audio_num = audio_num[:9]
    # print('%s without extension' %audio_num)
    out_dict[audio_num]=text_stitched
    line_num+=1
  # i=0
  # for keys, value in out_dict.items():
  #   print('%s : %s' %(keys,value))
  #   i+=1
  #   if i>5:
  #     break
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



def line_to_charset(line_dict, unclassified_label,jstart=91):
  extra_dict={}
  # j=91
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


def prepare_input_output(batch_size=32):
  file_path  = '/content/drive/My Drive/PartA_Gujarati/Train/Audio'
  file_folder = os.listdir(file_path)
  count=0
  x=[]
  y=[]
  for filez in file_folder:
    file_name = filez.split('.')[0]
    if file_name in out_dict:
      sample_rate, signal = scipy.io.wavfile.read(os.path.join(file_path,filez))
      x_data = convert_input(sample_rate, signal)
      x.append(x_data)
      y.append(out_dict[file_name])
      count+=1
    else:
      # print('not present')
      continue
    # if count>=batch_size:
    #   break
  # print(type(x))
  # print(type(y))
  # print(type(x[0]))
  # print(x[0].shape)
  # print(type(y[0]))
  y = line_to_charset(y,90)
  y = np.array(y)
  # print(type(y))
  # print(y.shape)
  return(x,y)
      # x=[]
      # y=[]
      # count=0
      # return (x,y)

def network(length_char_dict=65):
  length_char_dict=65
  input_layer = Input(name = 'input',shape=(None,80))
  bilstm0 = Bidirectional(LSTM(240,return_sequences=True))(input_layer)
  bilstm1 = Bidirectional(LSTM(240,return_sequences=True))(bilstm0)
  bilstm2 = Bidirectional(LSTM(240,return_sequences=True))(bilstm1)
  bilstm3 = Bidirectional(LSTM(240,return_sequences=True))(bilstm2)
  dense = TimeDistributed(Dense(length_char_dict))(bilstm3)
  output_layer = Activation('softmax', name='softmax')(dense)
  network_model = CTCModel([input_layer],[output_layer])
  network_model.compile(optimizer=Adam(lr=1e-3), metrics=['accuracy'])
  network_model.summary()
  x_train_len = np.asarray([len(x_train[i]) for i in range(len(x_train))])
  y_train_len = np.asarray([1 for i in range(len(y_train))])

def train_model(num_epochs=4):
  for i in range(num_epochs):
    history = network_model.fit(x=[x_train,y_train,x_train_len,y_train_len], y=np.zeros(len(x_train)), epochs = 1,
                             verbose=2, validation_split=0.15, batch_size=32)
    # loss.append(hist.history['loss'])
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['training', 'validation'], loc='upper left')
  plt.show()

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()

if __name__=='__main__':
           prepare_output()
           alpha = prepare_input_output(32)
           alpha = list(alpha)
           (x_train,y_train) = alpha[0]
           x_train = np.array(x_train)
           # y_test = alpha[1]
#            print(type(x_train))
#            print(type(y_train))
           network(len(char_dict))
           train_model(4000)
           
