import tensorflow as tf
from Face_detection_lovelyz.utils import *

# FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("image_width", 100, "width of image")
flags.DEFINE_integer("image_height", 100, "height of image")
flags.DEFINE_integer("image_channels", 3, "color of image")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("dropout", 0.7, "learning rate")
flags.DEFINE_float("beta1", 0.9, "beta1")

# input
image_list = list_image('C:/Users\Chulkyun Park\Documents\PycharmProjects\Face_detection_lovelyz\Test dataset')
print(image_list)
filenames = [ d['filename'] for d in image_list]
label_indexes = [ d['label_index'] for d in image_list]

one_hot = tf.one_hot(label_indexes, 3)
filename_queue, labelname_queue = tf.train.slice_input_producer([filenames, one_hot], shuffle=True)
images_and_labels = []
num_preprocess_threads = 4

# Threading and Queue
for thread_id in range(num_preprocess_threads):
    image_buffer = tf.read_file(filename_queue)
    decoded_image = tf.image.decode_jpeg(image_buffer, channels=3)
    converted_image = tf.image.convert_image_dtype(decoded_image, tf.float32) # convert image to [0,1)
    expanded_image = tf.expand_dims(converted_image, 0)
    reshaped_image = tf.image.resize_bilinear(expanded_image, [FLAGS.image_height, FLAGS.image_width], align_corners=False)
    images_and_labels.append([reshaped_image, labelname_queue])

# batch input
images, label_index_batch = tf.train.batch_join(images_and_labels, batch_size = FLAGS.batch_size , capacity = 2* num_preprocess_threads*FLAGS.batch_size)
cast_image = tf.cast(images, tf.float32)
batch_inputs = tf.reshape(cast_image, shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channels])
batch_labels = tf.reshape(label_index_batch, [FLAGS.batch_size, 3])

x_image = tf.reshape(batch_inputs, [-1, FLAGS.image_height, FLAGS.image_width, 3])


# Variables
weight_1 = tf.get_variable('weight_1',shape=[3, 3, 3, 32],initializer=tf.truncated_normal_initializer(stddev=0.02))
bias_1 = tf.get_variable('bias_1', shape=[32], initializer=tf.constant_initializer(0.0))

weight_2 = tf.get_variable('weight_2',shape=[3, 3, 32, 32],initializer=tf.truncated_normal_initializer(stddev=0.02))
bias_2 = tf.get_variable('bias_2', shape=[32], initializer=tf.constant_initializer(0.0))

weight_3 = tf.get_variable('weight_3',shape=[3, 3, 32, 64],initializer=tf.truncated_normal_initializer(stddev=0.02))
bias_3 = tf.get_variable('bias_3', shape=[64], initializer=tf.constant_initializer(0.0))

weight_4 = tf.get_variable('weight_4',shape=[5, 5, 64, 64],initializer=tf.truncated_normal_initializer(stddev=0.02))
bias_4 = tf.get_variable('bias_4', shape=[64], initializer=tf.constant_initializer(0.0))

FC_weight = tf.get_variable('FC_weight', [FLAGS.image_height * FLAGS.image_width * 64, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
FC_bias = tf.get_variable('FC_bias', [256], initializer=tf.constant_initializer(0.0))

output_weight = tf.get_variable('output_weight', [256, 3], initializer=tf.truncated_normal_initializer(stddev=0.02))
output_bias = tf.get_variable('output_bias', [3], initializer=tf.constant_initializer(0.0))

# Neural Network
h_1 = tf.nn.conv2d(x_image, weight_1, strides=[1,1,1,1], padding='SAME')
h_1_bias_add = tf.reshape(tf.nn.bias_add(h_1, bias_1), h_1.get_shape())
h_1_relu = tf.nn.relu(h_1_bias_add)

h_2 = tf.nn.conv2d(h_1_relu, weight_2, strides=[1,1,1,1], padding='SAME')
h_2_bias_add = tf.reshape(tf.nn.bias_add(h_2, bias_2), h_2.get_shape())
h_2_relu = tf.nn.relu(h_1_bias_add)

h_2_pool = tf.nn.max_pool(h_2_relu, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

h_3 = tf.nn.conv2d(h_2_pool, weight_3, strides=[1,1,1,1], padding='SAME')
h_3_bias_add = tf.reshape(tf.nn.bias_add(h_3, bias_3), h_3.get_shape())
h_3_relu = tf.nn.relu(h_3_bias_add)

h_4 = tf.nn.conv2d(h_3_relu, weight_4, strides=[1,1,1,1], padding='SAME')
h_4_bias_add = tf.reshape(tf.nn.bias_add(h_4, bias_4), h_4.get_shape())
h_4_relu = tf.nn.relu(h_4_bias_add)
h_4_pool = tf.nn.max_pool(h_4_relu, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
h_4_dropout = tf.nn.dropout(h_4_pool, 0.25)

h_flat = tf.reshape(h_4_dropout, [-1, FLAGS.image_height * FLAGS.image_width * 64])
h_FC_layer = tf.nn.relu(tf.matmul(h_flat, FC_weight) + FC_bias)
h_dropout_FC = tf.nn.dropout(h_FC_layer, 0.5)

h_FC_layer_2 = tf.matmul(h_dropout_FC, output_weight) + output_bias # [32,3]
print(h_FC_layer_2.get_shape())

# Cost function
entrophy = tf.nn.softmax_cross_entropy_with_logits(logits=h_FC_layer_2, labels=batch_labels)
cost = tf.reduce_mean(entrophy)
batch_labels = tf.cast(batch_labels, tf.float32)
ev_correct_prediction = tf.equal(tf.argmax(h_FC_layer_2,1), tf.argmax(batch_labels,1))
accuracy = tf.reduce_mean(tf.cast(ev_correct_prediction, tf.float32))

param_list = [weight_1, weight_2, weight_3, weight_4, bias_1, bias_2, bias_3, bias_4, FC_weight, FC_bias, output_weight, output_bias]
saver = tf.train.Saver(param_list)


# Optimzer
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1= FLAGS.beta1).minimize(cost)

# Summary
# image_summary = tf.summary.image
# histogram_summary = tf.summary.histogram
# merge_summary = tf.summary.merge
# scalar_summary = tf.summary.scalar
# File_summary = tf.summary.FileWriter

with tf.Session() as sess:
    saver.restore(sess, './saver/tensorflow.ckpt')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    for i in range(1000):
        if i % 10 == 0:
            print("---------------------")
            print("Epoch:", i)
            print("Cost:" , sess.run(cost))
            print("Accuracy:" , sess.run(accuracy))
    coord.request_stop()
    coord.join(threads)
