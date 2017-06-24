import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(0,len(series)-window_size):
        tempX = series[i:i+window_size].tolist()
        tempy = series[i+window_size].tolist()
        X.append(tempX)
        y.append(tempy)
        
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    import keras

    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)


    # TODO: build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1),return_sequences=False))
    model.add(Dense(units=1))


    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model



### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
  import string
  filter_list = []

  for letter in string.ascii_lowercase:
      filter_list.append(ord(letter))
    
  for punctuation in [' ', '!', ',', '.', ':', ';', '?']:
    filter_list.append(ord(punctuation))

  for c in text:
    if ord(c) not in filter_list:
        text = text.replace(c,' ')
    
  # shorten any extra dead space created above
  text = text.replace('  ',' ')

  return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    i = 0
    while i < len(text)-window_size:
        tempX = text[i:i+window_size]
        tempy = text[i+window_size]
        inputs.append(tempX)
        outputs.append(tempy)
        i += step_size
    
    return inputs,outputs
