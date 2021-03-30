import io
import pickle

"""
    TODO: This is a class for keeping track of, serializing and deserializing objects
"""


class PolicyManager(pickle.Pickler):
    obj_path = None

    def __init__(self, obj_path="./data/policy.pickle"):
        self.obj_path = obj_path

    """
        Save objects of Policy type only.
    """
    def save(self, obj):
        if obj.__class__.__name__[-6:] == "Policy":
            pickle.dump(obj, open(self.obj_path, "rb"))

        return

    def load(self):
        obj = pickle.load(open(self.obj_path, "rb"))
        if obj.__class__.__name__[-6:] != "Policy":
            obj = None

        return obj
