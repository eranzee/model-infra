import json


class DynamicObject:
    def __init__(self, path=None):
        if path is not None:
            self.load_json(path)
        else:
            self.internal_dict = {}

    def get_keys(self):
        return self.internal_dict.keys()

    def get_values(self):
        return self.internal_dict.values()

    def __getitem__(self, key):
        if key not in self.internal_dict.keys():
            return None

        if self.internal_dict[key] is None:
            self.internal_dict.pop(key)
            return None

        return self.internal_dict[key]

    def __setitem__(self, key, value):
        self.internal_dict[key] = value

    def load_json(self, path):
        with open(path) as json_file:
            self.internal_dict = json.load(json_file)
