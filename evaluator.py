import tensorflow as tf
from batch import batch_setup
from model import my_cnn
import numpy as np

class evaluate(object):
    def __init__(self, path_to_eval_log_dir):
        self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)

    def my_evaluate(self, checkpoint_path, tf_path, num_examples, global_step):
        batch_size = 128
        num_batches = int(np.ceil(num_examples / batch_size))
        needs_include_length = False

        with tf.Graph().as_default():
            img_batch, leng_batch, dig_batch = batch_setup.my_batch(tf_path,
                                                                              num_examples=num_examples,
                                                                              batch_size=batch_size,
                                                                              shuffled=False)
            leng_logits, dig_logits = my_cnn.cnn(img_batch, drop_rate=0.0)
            leng_predictions = tf.argmax(leng_logits, axis=1)
            dig_predictions = tf.argmax(dig_logits, axis=2)

            if needs_include_length:
                labels = tf.concat([tf.reshape(leng_batch, [-1, 1]), dig_batch], axis=1)
                predictions = tf.concat([tf.reshape(leng_predictions, [-1, 1]), dig_predictions], axis=1)
            else:
                labels = dig_batch
                predictions = dig_predictions

            labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
            predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)

            accuracy, new_accuracy = tf.metrics.accuracy(
                labels=labels_string,
                predictions=predictions_string
            )

            tf.summary.image('image', img_batch)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram('variables',tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))
            summary = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                restorer = tf.train.Saver()
                restorer.restore(sess, checkpoint_path)

                for _ in range(num_batches):
                    sess.run(new_accuracy)

                accuracy_val, summary_val = sess.run([accuracy, summary])
                self.summary_writer.add_summary(summary_val, global_step=global_step)

                coord.request_stop()
                coord.join(threads)

        return accuracy_val
