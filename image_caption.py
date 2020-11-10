import numpy as np   
import pandas as pd   
import os 
import tensorflow as tf 
from keras.preprocessing.sequence import pad_sequences 
from keras.preprocessing.text import Tokenizer 
from keras.models import Model 
from keras.layers import Flatten, Dense, LSTM, Dropout, Embedding, Activation 
from keras.layers import concatenate, BatchNormalization, Input
from keras.layers.merge import add 
from keras.utils import to_categorical, plot_model 
from keras.applications.inception_v3 import InceptionV3, preprocess_input 
import matplotlib.pyplot as plt
import cv2 
import string
import pickle



# Load Descriptions
def load_description(text): 
    mapping = dict() 
    for line in text.split("\n"): 
        token = line.split("\t") 
        if len(line) < 2:   # remove short descriptions 
            continue
        image_id = token[0].split('.')[0] # name of the image 
        image_desc = token[1]              # description of the image 
        if img_id not in mapping: 
            mapping[image_id] = list() 
        mapping[image_id].append(image_desc) 
    return mapping 
  
token_path = 'Flickr_Data / Flickr_TextData / Flickr8k.token.txt'
text = open(token_path, 'r', encoding = 'utf-8').read() 
descriptions = load_description(text) 



# Cleaning the Text
def clean_descriptions(descriptions): 
    for key, desc_list in descriptions.items(): 
        for i in range(len(desc_list)): 
            desc = desc_list[i] 
            desc = [ch for ch in desc if ch not in string.punctuation] 
            desc = ''.join(desc) 
            desc = desc.split(' ') 
            desc = [word.lower() for word in desc if len(word)>1 and word.isalpha()] 
            desc = ' '.join(desc) 
            desc_list[i] = desc
  
clean_description(descriptions) 



# Generate the Vocabulary
def to_vocab(descriptions): 
    words = set() 
    for key in desc.keys(): 
        for line in desc[key]: 
            words.update(line.split()) 
    return words 
vocab = to_vocab(descriptions)



# Load the Images
import glob 
images = 'Flickr_Data/Images/'
# Create a list of all image names in the directory 
images_path = glob.glob(images + '*.jpg') 
  
train_path = 'Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'
train_images = open(train_path, 'r', encoding = 'utf-8').read().split("\n") 
train_images_path = []  # list of all images in training set 
for path in images_path: 
    if(path[len(images):] in train_images): 
        train_images_path.append(path) 
          
# load descriptions of training set in a dictionary. Name of the image will act as ey 
def load_clean_descriptions(descriptions, dataset): 
    dataset_desc = dict() 
    for key, desc_list in descriptions.items(): 
        if key+'.jpg' in dataset: 
            if key not in dataset_desc: 
                dataset_desc[key] = list() 
            for line in desc_list: 
                desc = 'startseq ' + line + ' endseq'
                dataset_des[key].append(desc) 
    return dataset_desc
  
train_descriptions = load_clean_descriptions(descriptions, train_images) 



# Extract the feature vector from all images
from keras.preprocessing.image import load_img, img_to_array 
def preprocess_img(image_path): 
    # inception v3 excepts img in 299 * 299 * 3 
    img = load_img(image_path, target_size = (299, 299)) 
    x = img_to_array(img) 
    # Add one more dimension 
    x = np.expand_dims(x, axis = 0) 
    x = preprocess_input(x) 
    return x 
  
def encode(image): 
    image = preprocess_img(image) 
    vec = model.predict(image) 
    vec = np.reshape(vec, (vec.shape[1])) 
    return vec 
  
base_model = InceptionV3(weights = 'imagenet') 
model = Model(base_model.input, base_model.layers[-2].output) 
# run the encode function on all train images and store the feature vectors in a list 
encoding_train = {} 
for img in train_img: 
    encoding_train[img[len(images):]] = encode(img)

# Save the bottleneck train features to disk
with open("encoded_train_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)



# Tokenizing the vocabulary
# list of all training captions 
all_train_captions = [] 
for key, desc_list in train_descriptions.items(): 
    for desc in desc_list: 
        all_train_captions.append(desc) 
  
# consider only words which occur atleast 10 times 
vocabulary = vocab 
threshold = 10 # you can change this value according to your need 
word_counts = {} 
for desc in all_train_captions: 
    for word in desc.split(' '): 
        word_counts[word] = word_counts.get(word, 0) + 1
  
vocab = [word for word in word_counts if word_counts[word] >= threshold] 
  
# word mapping to integers 
ixtoword = {} 
wordtoix = {} 
  
ix = 1
for word in vocab: 
    wordtoix[word] = ix 
    ixtoword[ix] = word 
    ix += 1
      
# find the maximum length of a description in a dataset 
max_length = max(len(desc.split()) for desc in all_train_captions)



# Glove vector embeddings
X1, X2, y = list(), list(), list() 
for key, des_list in train_descriptions.items(): 
    pic = train_features[key + '.jpg'] 
    for cap in des_list: 
        seq = [wordtoix[word] for word in cap.split(' ') if word in wordtoix] 
        for i in range(1, len(seq)): 
            in_seq, out_seq = seq[:i], seq[i] 
            in_seq = pad_sequences([in_seq], maxlen = max_length)[0] 
            out_seq = to_categorical([out_seq], num_classes = vocab_size)[0] 
            # store 
            X1.append(pic) 
            X2.append(in_seq) 
            y.append(out_seq) 
  
X2 = np.array(X2) 
X1 = np.array(X1) 
y = np.array(y) 
  
# load glove vectors for embedding layer 
embeddings_index = {} 
golve_path ='glove.6B.200d.txt'
glove = open(golve_path, 'r', encoding = 'utf-8').read() 
for line in glove.split("\n"): 
    values = line.split(" ") 
    word = values[0] 
    indices = np.asarray(values[1: ], dtype = 'float32') 
    embeddings_index[word] = indices 
  
emb_dim = 200
emb_matrix = np.zeros((vocab_size, emb_dim)) 
for word, i in wordtoix.items(): 
    emb_vec = embeddings_index.get(word) 
    if emb_vec is not None: 
        emb_matrix[i] = emb_vec 



# Define the Model 
ip1 = Input(shape = (2048, )) 
fe1 = Dropout(0.2)(ip1) 
fe2 = Dense(256, activation = 'relu')(fe1) 
ip2 = Input(shape = (max_length, )) 
se1 = Embedding(vocab_size, emb_dim, mask_zero = True)(ip2) 
se2 = Dropout(0.2)(se1) 
se3 = LSTM(256)(se2) 
decoder1 = add([fe2, se3]) 
decoder2 = Dense(256, activation = 'relu')(decoder1) 
outputs = Dense(vocab_size, activation = 'softmax')(decoder2) 
model = Model(inputs = [ip1, ip2], outputs = outputs)



# Training the Model
model.layers[2].set_weights([emb_matrix]) 
model.layers[2].trainable = False
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam') 
model.fit([X1, X2], y, epochs = 50, batch_size = 256) 



# Predicting the Output
def predict_caption(pic): 
    start = 'startseq'
    for i in range(max_length): 
        seq = [wordtoix[word] for word in start.split() if word in wordtoix] 
        seq = pad_sequences([seq], maxlen = max_length) 
        yhat = model.predict([pic, seq]) 
        yhat = np.argmax(yhat) 
        word = ixtoword[yhat] 
        start += ' ' + word 
        if word == 'endseq': 
            break
    final = start.split() 
    final = final[1:-1] 
    final = ' '.join(final) 
    return final