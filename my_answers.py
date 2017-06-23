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
    # find all unique characters in the text

    unique = list((set(text)))
    decimal = [ord(c) for c in unique]
    comb = {k:v for k in unique for v in decimal}
    a = dict(zip(unique,decimal))

    #Prints unique characters along with decimal equivalent, useful for characters which do not have any direct corresponding key
    print(a)

    # remove as many non-english characters and character sequences as you can

    #Result of the print can be seen in Jupyter notebook. This is what we get. 

    '''
    {' ': 32,
     '!': 33,
     '"': 34,
     '$': 36,---> Remove
     '%': 37,---> Remove
     '&': 38,
     "'": 39,
     '(': 40,
     ')': 41,
     '*': 42,
     ',': 44,
     '-': 45,---> Remove
     '.': 46,
     '/': 47,
     '0': 48,
     '1': 49,
     '2': 50,
     '3': 51,
     '4': 52,
     '5': 53,
     '6': 54,
     '7': 55,
     '8': 56,
     '9': 57,
     ':': 58,
     ';': 59,
     '?': 63,
     '@': 64,---> Remove
     'a': 97,
     'b': 98,
     'c': 99,
     'd': 100,
     'e': 101,
     'f': 102,
     'g': 103,
     'h': 104,
     'i': 105,
     'j': 106,
     'k': 107,
     'l': 108,
     'm': 109,
     'n': 110,
     'o': 111,
     'p': 112,
     'q': 113,
     'r': 114,
     's': 115,
     't': 116,
     'u': 117,
     'v': 118,
     'w': 119,
     'x': 120,
     'y': 121,
     'z': 122,
     'à': 224,---> Remove
     'â': 226,---> Remove
     'è': 232,----> Remove
     'é': 233}----> Remove
    '''
    filter_list = [36,37,38,64,45,224,226,232,233]
    
    for c in filter_list:
        text = text.replace(chr(c),' ')
    
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
