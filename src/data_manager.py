import pandas as pd

class Data_Manager:
    def __init__(self, file):
        self._data_set = self._read_data_set_from_file(file)
        
    def _read_data_set_from_file(self, file):
        # Replace 'path_to_your_data_file' with the actual file path
        return pd.read_csv(file, sep='\t', header=None, names=['Label', 'Message'])
    
    def get_data_set(self):
        return self._data_set
    
    def get_data_set_head(self):
        return self._data_set.head()