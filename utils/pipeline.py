from imageprocessor import get_frames
import numpy as np
import h5py
from utilfunctions import print_progress
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, GlobalAveragePooling2D, Flatten

def create_model(model,input_shape): 
    cnn_model = model(input_shape=input_shape, weights='imagenet', include_top=False)
    cnn_model.trainable = False
    print(cnn_model.summary())
    sequential_layers = [cnn_model,GlobalAveragePooling2D()]
    cnn_model = Sequential(sequential_layers,name=cnn_model.name)
    return cnn_model

#function that processes all videos and yiels its values and labels 
def proces_transfer(vid_names, in_dir, labels,cnn_model,img_size,frames_per_file):
    
    count = 0
    
    tam = len(vid_names)
    
    while count<tam:
        
        video_name = vid_names[count]
        
        image_batch = get_frames(in_dir, video_name,img_size, frames_per_file)
        
        transfer_values = cnn_model.predict(image_batch)
         
        labels1 = labels[count]
        
        aux = np.ones([frames_per_file,2])
        
        labelss = labels1*aux
        yield transfer_values, labelss
        
        count+=1

#uses proces_transfer and gets each of the images and its frames and labels and write it to a h5py file
def make_files(n_files,names,labels,image_model,output_file,frames_per_file,img_size,in_dir="data"):
    
    gen = proces_transfer(names, in_dir, labels,image_model,img_size, frames_per_file)

    numer = 1

    # Read the first chunk to get the column dtypes
    chunk = next(gen)
    
    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]
    with h5py.File(output_file, 'w') as f:
    
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
    
    
        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
    
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)
    
         # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            
            if numer == n_files:
            
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]
            
            print_progress(numer, n_files)
        
            numer += 1

def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)  # Round to 0 or 1
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

def getLSTM(cells,chunk_size,n_chunks):
    model = Sequential()
    model.add(Bidirectional(LSTM(cells, input_shape=(n_chunks, chunk_size))))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy',
                                                                        #f1_score
                                                                        ])
    return model