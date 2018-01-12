import os
import time
import tensorflow as tf
from data import data_io
from batch import batch_setup
from model import my_cnn
from evaluator import evaluate


def my_training(train_path, 
                num_train, 
                val_path, num_val,
                log_path, 
                checkpoint_path, 
                training_options):
    
    batch_size = training_options['batch_size']
#    initial_patience = training_options['patience']
    steps_loss = 100
    steps_check = 1000

    with tf.Graph().as_default():
        img_batch, leng_batch, dig_batch = batch_setup.my_batch(train_path,
                                                                     num_examples=num_train,
                                                                     batch_size=batch_size,
                                                                     shuffled=True)
        leng, dig = my_cnn.cnn(img_batch, drop_rate=0.2)
        loss = my_cnn.loss(leng, dig, leng_batch, dig_batch)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(training_options['learning_rate'], global_step=global_step,
                                                   decay_steps=training_options['decay_steps'], decay_rate=training_options['decay_rate'], staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.image('image', img_batch)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(log_path, sess.graph)
            evaluator = evaluate(os.path.join(log_path, 'eval/val'))

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver = tf.train.Saver()
            if checkpoint_path is not None:
                assert tf.train.checkpoint_exists(checkpoint_path), \
                    '%s not found' % checkpoint_path
                saver.restore(sess, checkpoint_path)
                print ('Load the model from: %s' % checkpoint_path)
            
            print("Building my CNN:")
            print("Parameters:")
            print("\tBatch size: {}".format(training_options['batch_size']))
            print("\tLearning rate: {}".format(training_options['learning_rate']))
            print("\tInitial patience: {}".format(training_options['patience']))
            print("\tDecay steps:{}".format(training_options['decay_steps']))
            print("\tDecay rate:{}".format(training_options['decay_rate']))
            print ('Start training') 
            
#            patience = initial_patience
            best_accuracy = 0.0
            duration = 0.0

            while True:
                start_time = time.time()
                _, loss_val, summary_val, global_step_val, learning_rate_val = sess.run([train_op, loss, summary, global_step, learning_rate])
                duration += time.time() - start_time

                if global_step_val % steps_loss == 0:
                    examples_per_sec = batch_size * steps_loss / duration
                    duration = 0.0
                    print('Step {:d}, loss = {:f} ({:.1f} examples/sec)'.format(
                        global_step_val, loss_val, examples_per_sec))
                if global_step_val % steps_check != 0:
                    continue

                writer.add_summary(summary_val, global_step=global_step_val)

                print ('Validation:')
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(log_path, 'latest.ckpt'))
                accuracy = evaluator.my_evaluate(path_to_latest_checkpoint_file, val_path,
                                              num_val,
                                              global_step_val)
                print ('Accuracy = {:f}, best accuracy {:f}'.format(accuracy, best_accuracy))

                if accuracy > best_accuracy:
                    checkpoint_file = saver.save(sess, os.path.join(log_path, 'model.ckpt'),
                                                         global_step=global_step_val)
                    print('Best validation accuracy! accuracy: {}'.format(accuracy))
                    print ('Model saved to file: {:s}'.format(checkpoint_file))
#                    patience = initial_patience
                    best_accuracy = accuracy
#                else:
#                    patience -= 1

#                print ('Patience = {:d}\n'.format(patience))
#                if patience == 0:
#                    break
                
                if global_step_val > 50000:
                    print("Global step value reaches {}, training is stopped\n".format(global_step_val))
                    break

            coord.request_stop()
            coord.join(threads)
#            print ('Finished')
            print("Traning ends. The best valid accuracy is {}.".format(best_accuracy))


# Use the commends below to run the function in jupyter notebook
# flags={ 
#         'data_dir' : './data', 
#         'train_logdir' : './logs/train1',
#         'restore_checkpoint' : None,
#         'batch_size' : 32,
#         'learning_rate' : 1e-2, 
#         'patience' : 100, 
#         'decay_steps' : 10000,
#         'decay_rate' : 0.9
#         }

# train_path = os.path.join(flags['data_dir'], 'train.tfrecords')
# val_path = os.path.join(flags['data_dir'], 'val.tfrecords')
# meta_file = os.path.join(flags['data_dir'], 'meta.json')
# train_log = flags['train_logdir']
# ckpt_file = flags['restore_checkpoint']
# training_options = {
#                     'batch_size': flags['batch_size'],
#                     'learning_rate': flags['learning_rate'],
#                     'patience': flags['patience'],
#                     'decay_steps': flags['decay_steps'],
#                     'decay_rate': flags['decay_rate']
#                     }

# data = data_io()
# data.load(meta_file)

# my_training(
#             train_path, data.num_train,
#             val_path, data.num_val,
#             train_log,ckpt_file,
#             training_options
#             )
