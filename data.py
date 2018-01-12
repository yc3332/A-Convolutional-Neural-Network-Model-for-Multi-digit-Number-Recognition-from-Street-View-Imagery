import json


class data_io(object):
    def __init__(self):
        self.num_train = None
        self.num_val = None
        self.num_test = None

    def save(self, path_to_json_file):
        with open(path_to_json_file, 'w') as f:
            content = {
                'num_examples': {
                    'train': self.num_train,
                    'val': self.num_val,
                    'test': self.num_test
                }
            }
            json.dump(content, f)

    def load(self, path_to_json_file):
        with open(path_to_json_file, 'r') as f:
            content = json.load(f)
            self.num_train = content['num_examples']['train']
            self.num_val = content['num_examples']['val']
            self.num_test = content['num_examples']['test']
