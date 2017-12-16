############################################################
#                                                          #
#  Code for Lab 3: Data Augmentation and Debugging Strat.  #
#                                                          #
############################################################

"""Based on TensorFLow's tutorial: A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf
import batch_generator as bg

import numpy as np
# import _pickle as pickle
import cPickle as pickle

data = pickle.load(open('dataset.pkl','rb'))
# data = open('dataset.pkl','rb')


here = os.path.dirname(__file__)
sys.path.append(here)
sys.path.append(os.path.join(here, '..', 'CIFAR10'))


#(train_images,train_labels) = gtsrb.batch_generator(data, 'train').next()
#print(train_images)
#print(train_labels)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries. (default: %(default)d)')
tf.app.flags.DEFINE_integer('flush-frequency', 50,
                            'Number of steps between flushing summary results. (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model-frequency', 100,
                            'Number of steps between model saves. (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('max-steps', 500,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', 100, 'Number of examples per mini-batch. (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-3, 'Number of examples to run. (default: %(default)d)')

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3
CLASS_COUNT = 43

weight_decay = 0.0005

run_log_dir = os.path.join(FLAGS.log_dir, 'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                                       lr=FLAGS.learning_rate))
checkpoint_path = os.path.join(run_log_dir, 'model.ckpt')

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)


def deepnn(x_image, img_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), class_count=CLASS_COUNT):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.

    Args:
        x_image: an input tensor whose ``shape[1:] = img_space``
            (i.e. a batch of images conforming to the shape specified in ``img_shape``)
        img_shape: Input image shape: (width, height, depth)
        class_count: number of classes in dataset

    Returns: A tensor of shape (N_examples, 10), with values equal to the logits of
      classifying the object images into one of 10 classes
      (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
    """

    # First convolutional layer - maps one RGB image to 32 feature maps.
    #2
    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        use_bias=False,
        name='conv1'
    )
    #3
    pool1 = tf.layers.average_pooling2d(
        inputs=conv1,
        pool_size=[3, 3],
        strides=2,
        padding="same",
        name='pool1'
    )
    #4
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        use_bias=False,
        name='conv2'
    )
    #5
    pool2 = tf.layers.average_pooling2d(
        inputs=conv2,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool2'
    )
    #6
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        use_bias=False,
        name='conv3'
    )
    #7
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool3'
    )
    #8
    pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 64], name='pool3_flattened')
    fc1 = tf.layers.dense(inputs=pool3_flat, units=64, name='fc1')
    #9
    logits = tf.layers.dense(inputs=fc1, units=class_count, name='fc2')


    return logits


def main(_):
    tf.reset_default_graph()

    trainGenerator = bg.batch_generator(data, 'train');
    testGenerator = bg.batch_generator(data, 'test');



    # Build the graph for the deep net
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS])
        x_image = tf.reshape(x, [-1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
        #whitening
        #x_image = tf.map_fn(lambda image: tf.image.per_image_standardization(image), x_image)
        y_ = tf.placeholder(tf.float32, [None, CLASS_COUNT])


    with tf.name_scope('model'):
        y_conv = deepnn(x_image)


    #   # Create your variables
    # weights = tf.get_variable('weights', collections=['variables'])
    # # W = tf.get_variable(name='weight', shape=x_image, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    #   #
    #   #s
    # with tf.variable_scope('weights_norm') as scope:
    #     weights_norm = tf.reduce_sum(
    #       input_tensor = 0.0005*tf.pack(
    #           [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
    #       ),
    #       name='weights_norm'
    #     )

    #   reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #   reg_constant = 0.01  # Choose an appropriate one.
    #   loss = my_normal_loss + reg_constant * sum(reg_losses)

    # Add the weight decay loss to another collection called losses
    # tf.add_to_collection('losses', weights_norm)

    # Add the other loss components to the collection losses
    # tf.add_to_collection('losses', cross_entropy)

    #   To calculate your total loss



    # weights = tf.get_variable(
    #     name="weights",
    #     regularizer=regularizer
    # )

    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # reg_constant = 0.0005  # Choose an appropriate one.


    # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES);
    # prinf(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))



    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)) #+ weights_norm

    # tf.add_to_collection('losses', cross_entropy)

    # cross_entropy = cross_entropy + reg_losses
    # tf.add_n(tf.get_collection('losses'), name='total_loss')

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow
    # decay_steps = 10000000  # decay the learning rate every 100000 steps
    # decay_rate = 0.8  # the base of our exponential for the decay
    # decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                            #    decay_steps, decay_rate, staircase=True)

    # We need to update the dependencies of the minimization op so that it all ops in the `UPDATE_OPS`
    # are added as a dependency, this ensures that we update the mean and variance of the batch normalisation
    # layers
    # See https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization for more
    trainVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        # weight_decay = tf.constant(0.0005, dtype=tf.float32) # your weight decay rate, must be a scalar tensor.
        # W = tf.get_variable(name='weight', shape=x_image, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

        # train_step = tf.train.AdamOptimizer(decayed_learning_rate).minimize(cross_entropy, global_step=global_step)

        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
        # tf.contrib.layers.apply_regularization(regularizer, trainVariables)
        train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9).minimize(cross_entropy,global_step=global_step)


    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    learning_rate_summary = tf.summary.scalar("Learning Rate", FLAGS.learning_rate)
    img_summary = tf.summary.image('input images', x_image)

    train_summary = tf.summary.merge([loss_summary, accuracy_summary, learning_rate_summary, img_summary])
    validation_summary = tf.summary.merge([loss_summary, accuracy_summary])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(run_log_dir + "_validation", sess.graph)

        sess.run(tf.global_variables_initializer())

        # Training and validation
        for step in range(FLAGS.max_steps):
            #(trainImages, trainLabels) = cifar.getTrainBatch()
            #(testImages, testLabels) = cifar.getTestBatch()

            (trainImages, trainLabels) = bg.batch_generator(data,'train').next()
            (testImages, testLabels) = bg.batch_generator(data,'test').next()

            # tf.map_fn(lambda image: tf.image.per_image_standardization(image), trainImages)
            trainImages = tf.map_fn(lambda image: tf.image.random_brightness(image, max_delta=63), trainImages)
            # tf.image.per_image_standardization(trainImages)
            _, train_summary_str = sess.run([train_step, train_summary],
                                      feed_dict={x_image: trainImages, y_: trainLabels})

            # print(trainVariables)


            # Validation: Monitoring accuracy using validation set
            if (step + 1) % FLAGS.log_frequency == 0:
                validation_accuracy, validation_summary_str = sess.run([accuracy, validation_summary],
                                                                       feed_dict={x_image: testImages, y_: testLabels})
                print('step {}, accuracy on validation set : {}'.format(step, validation_accuracy))
                train_writer.add_summary(train_summary_str, step)
                validation_writer.add_summary(validation_summary_str, step)


            # Save the model checkpoint periodically.
            if (step + 1) % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, checkpoint_path, global_step=step)

            if (step + 1) % FLAGS.flush_frequency == 0:
                train_writer.flush()
                validation_writer.flush()

        # Resetting the internal batch indexes
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0

        nTestSamples = 12630
        # testGenerator = gb.batch_generator(data, 'test');

        # while evaluated_images != 12630:
        for (testImages, testLabels) in bg.batch_generator(data, 'test'):
            # Don't loop back when we reach the end of the test set
            #(testImages, testLabels) = cifar.getTestBatch(allowSmallerBatches=True)
            # (testImages, testLabels) = testGenerator.next()

            test_accuracy_temp = sess.run(accuracy, feed_dict={x_image: testImages, y_: testLabels})

            batch_count += 1
            test_accuracy += test_accuracy_temp
            # evaluated_images += testLabels.shape[0]

        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)
        print('model saved to ' + checkpoint_path)

        train_writer.close()
        validation_writer.close()

if __name__ == '__main__':
    tf.app.run(main=main)
