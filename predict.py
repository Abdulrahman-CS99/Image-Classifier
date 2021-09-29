import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import argparse
from PIL import Image
import warnings
import pdb
import scipy
warnings.filterwarnings('ignore')



def process_image(image):
  image=tf.convert_to_tensor(image,tf.float32)
  image=tf.image.resize(image,(image_size,image_size))
  image/=255
  return image



def predict(img_path=None, model=None, k=None):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_image = process_image(test_image)    
    processed_image=np.expand_dims(processed_image,0)
    probs=model.predict(processed_image)
    return tf.nn.top_k(probs, k=k)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='myModel/test/1/image_06743.jpg') # use a deafault filepath to a primrose image
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()



def load_json(name):
    with open(name, 'r') as f:
        class_names = json.load(f)
    return class_names


def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image=tf.image.resize(image,(image_size,image_size))
    image /= 255
    return image, label

def loading_data():

    dataset = tfds.load('oxford_flowers102', shuffle_files=True, as_supervised = True, with_info = False)
    training_set, test_set, validation_set = dataset['train'], dataset['test'], dataset['validation']
    num_training_examples = dataset_info.splits['train'].num_examples
    return training_set, test_set, valid_set, training_set, num_training_examples

def batch_data(training_set, test_set, valid_set, num_training_examples):
    training_batches = training_set.cache().shuffle(num_training_examples//4).map(normalize(image, label)).batch(batch_size)
    test_batches = test_set.cache().map(normalize).batch(batch_size)
    valid_batches = valid_set.cache().map(normalize).batch(batch_size)
    return training_batches, test_batches, valid_batches

def adjustment(classes=None,class_names=None):
    return [class_names.get(str(key+1)) if key else "Placeholder" for key in classes.numpy().squeeze().tolist()]

def main():
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    args = parse_args()
    class_names=load_json(args.category_names)    
    
    feature_extractor = hub.KerasLayer(URL, input_shape=(224, 224, 3))
    model = tf.keras.Sequential([feature_extractor,
                                 tf.keras.layers.Dense(800,activation='relu'),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(400,activation='relu'),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(num_classes, activation='softmax')
                                ])
    
    
       
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(args.model_path)
    probs, classes = predict(image_path=args.img_path, model=model, top_k=args.top_k)
    pred_dict={filtered(classes,class_names)[i]: probs[0][i].numpy() for i in range(len(filtered(classes,class_names)))} 
    return probs, classes,filtered(classes,class_names),pred_dict    
    
if __name__ == '__main__':
    main()


