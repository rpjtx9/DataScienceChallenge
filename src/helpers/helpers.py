import os

file_path = os.path.dirname(__file__)
root_path = os.path.dirname((os.path.dirname(file_path)))
data_path = os.path.join(root_path, "data")