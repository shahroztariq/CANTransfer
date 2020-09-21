from keras import applications, Sequential, utils
from keras.layers import Dense, BatchNormalization, Dropout, regularizers, TimeDistributed, LSTM, Flatten, Conv2D, MaxPooling2D, ConvLSTM2D, Conv3D, Activation
from keras.regularizers import L1L2

# choose a number of time steps
n_steps = 16
n_features = 11
n_seq = 4
n_substeps = 4
batch_size=128
num_classes=2
epochs=50

def convlstm():
    L1L2(l1=0.0001, l2=0.0001)
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3,3), padding='same',activation='relu',
                         return_sequences=True, input_shape=(n_seq, 1, n_substeps, n_features),kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(filters=80, kernel_size=(3,3),padding='same',activation='relu', return_sequences=True,kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(ConvLSTM2D(filters=120, kernel_size=(3,3),padding='same',activation='relu',return_sequences=True,kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(filters=160, kernel_size=(3, 3), padding='same',activation='relu',return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(ConvLSTM2D(filters=200, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(filters=240, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same', data_format='channels_last', kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', padding='same', data_format='channels_last', kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model
