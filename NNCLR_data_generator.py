import tensorflow as tf
import numpy as np
import os
from contextlib import nullcontext
from random import shuffle
import tensorflow_addons as tfa

class SpikeDataset():
	def __init__(self, batch_size=32, shuffle_buffer=128, label_strides=(32,32), feature_len=128):

		self.snap_timespan = 128
		self.target_channel_num = 64
		self._shuffle_buffer = shuffle_buffer
		self._batch_size = batch_size
		self._phase = 0
		self._label_strides = label_strides
		self._one_hot = False
		self._feature_len = feature_len

	def create_flow_map(self, x, fr_multiplier=2, amplitude=2):
		f = np.random.normal(size=(self._batch_size,self.snap_timespan,self.target_channel_num,2)) * 3
		f[:,:,:,0] = 0

		for i in range(self.snap_timespan):
			k = np.sin(self._phase*fr_multiplier)*amplitude 
			f[:,i,:,1] = k
			self._phase = self._phase + 1
		
		return tfa.image.dense_image_warp(x, f)
		
	def prepare_files(self, file_paths):
		gt_files = []
		for folder in file_paths:
			tf_files = os.listdir(folder)
			gt_files.extend([os.path.join(folder, x) for x in tf_files if x.endswith("gt.tfrecord")])
							
		return gt_files

	def parser_fn(self, proto):

		features = {'template': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.string)}

		parsed_features = tf.io.parse_single_example(proto, features)
		images = tf.io.decode_raw(parsed_features['template'], tf.float32)
		labels = tf.io.decode_raw(parsed_features['label'], tf.float32)
				
		labels = tf.reshape(labels, (self.snap_timespan, -1, 1))
		images = tf.reshape(images, (self.snap_timespan, -1, 1))
		
		images = tf.image.resize_with_crop_or_pad(images, self.snap_timespan, self.target_channel_num)
		labels = tf.image.resize_with_crop_or_pad(labels, self.snap_timespan, self.target_channel_num)
		labels_bin = tf.cast(labels, tf.bool)
		labels_bin = tf.cast(labels_bin, tf.float32)
		images = images * labels_bin

		labels = tf.expand_dims(labels, axis=0)
		labels = tf.nn.max_pool2d(labels, self._label_strides, strides=self._label_strides, padding="SAME")
		labels = tf.reshape(labels, [-1,])
		
		if self._one_hot:
			labels = tf.cast(labels, tf.int32)
			labels = tf.one_hot(labels, self._feature_len)
		

		return images, labels

	def interleave_fn(self, filename):
		dataset = tf.data.TFRecordDataset(filename)
		dataset = dataset.map(self.parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		return dataset

	def create_sg_dataset(self, file_path, file_path2="", noise1=0, noise2=0):

		if file_path2 != "":
			dataset2 = self.create_sg_dataset(file_path2, noise1=noise2)
			
		self._noise = noise1
				
		file_list = self.prepare_files(file_path)
		dataset = tf.data.Dataset.from_tensor_slices(file_list)
		dataset = dataset.interleave(self.interleave_fn, cycle_length=10, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		dataset = dataset.repeat()

		dataset = dataset.shuffle(buffer_size=self._shuffle_buffer, reshuffle_each_iteration=True) # for proper training 10000
		dataset = dataset.batch(self._batch_size, drop_remainder=True)
		dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
			
		if file_path2 != "":
			dataset = tf.data.Dataset.zip((dataset, dataset2))

		return dataset