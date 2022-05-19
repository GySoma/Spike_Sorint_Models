import os
import random
import numpy as np
import tensorflow as tf

def create_sg_dataset(file_list,noise_level=0,infinite_dataset=True,labels_complex=True,finetune=False,max_pool=(8,8),neurons=75):
  
  def parser_fn(proto):
    
    features = {'template': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.string)}

    parsed_features = tf.io.parse_single_example(proto, features)
    images = tf.io.decode_raw(parsed_features['template'], tf.float32)
    labels = tf.io.decode_raw(parsed_features['label'], tf.float32)
        
    labels = tf.reshape(labels, (128, -1,1))
    images = tf.reshape(images, (128, -1,1))

    images = tf.image.resize_with_crop_or_pad(images, 128, 64)
    labels = tf.image.resize_with_crop_or_pad(labels, 128, 64)

    noise = np.random.normal(0,noise_level,8192)
    noise = np.reshape(noise, (128,64, 1))
    noise = tf.convert_to_tensor(noise,dtype=tf.float32)
    noise_images = images + noise
    
    labels_bin = tf.cast(labels, tf.bool)
    labels_bin = tf.cast(labels_bin, tf.float32)
    
    if labels_complex:
      labels = images*labels_bin

    if finetune:
      labels = tf.expand_dims(labels, axis=0)
      labels = tf.nn.max_pool(labels, max_pool, strides=max_pool, padding="SAME", data_format="NHWC")
      labels = tf.reshape(labels, [-1,])
      labels = tf.cast(labels, tf.int32)
      labels = tf.one_hot(labels, neurons)
    
    return (noise_images,labels)


  def interleave_fn(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset
    
  dataset = tf.data.Dataset.from_tensor_slices(file_list)
  dataset = dataset.interleave(interleave_fn, cycle_length=10, block_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(32)
  
  if infinite_dataset:
    dataset = dataset.shuffle(200)
    dataset = dataset.repeat()

  return dataset
  
def list_maker(path):
    
    final_list = []
    for i in path:
        final_list.extend(os.listdir(i))
        final_list = [os.path.join(i, x) for x in final_list]
        random.shuffle(final_list)
    return final_list
