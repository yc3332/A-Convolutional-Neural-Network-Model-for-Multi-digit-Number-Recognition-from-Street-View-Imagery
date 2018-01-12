import os
import tensorflow as tf
from data import data_io
from evaluator import evaluate

def my_test(path_to_checkpoint_dir, path_to_eval_tfrecords_file, num_eval_examples, path_to_eval_log_dir):
    evaluator = evaluate(path_to_eval_log_dir)
    print(path_to_checkpoint_dir)
    checkpoint_paths = tf.train.get_checkpoint_state(path_to_checkpoint_dir).all_model_checkpoint_paths
    for global_step, path_to_checkpoint in [(path.split('-')[-1], path) for path in checkpoint_paths]:
        try:
            global_step_val = int(global_step)
        except ValueError:
            continue

        accuracy = evaluator.my_evaluate(path_to_checkpoint, path_to_eval_tfrecords_file, num_eval_examples,
                                      global_step_val)
        print ('Evaluate {} on {}, accuracy = {}'.format(path_to_checkpoint, path_to_eval_tfrecords_file, accuracy))

# Use the commends below to run the function in jupyter notebook
# flags={ 'data_dir' : './data', 
#         'checkpoint_dir' : './logs/train', 
#         'eval_logdir' : './logs/eval'
#         }

# test_path = os.path.join(flags['data_dir'], 'test.tfrecords')
# meta_path = os.path.join(flags['data_dir'], 'meta.json')
# checkpoint_path = flags['checkpoint_dir']
# path_to_test_eval_log_dir = os.path.join(flags['eval_logdir'], 'test')

# data = data_io()
# data.load(meta_path)
# my_test(
#         checkpoint_path, 
#         test_path, 
#         data.num_test_examples, 
#         path_to_test_eval_log_dir
#         )