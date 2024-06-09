import json
import pickle

def get_json_data(file_name):
    with open(f'data/json/{file_name}.json', encoding='utf-8') as f:
        return json.load(f)
    
def get_image_path(file_name):
    return f'data/img/{file_name}'

def get_pickle_data(file_name):
    with open(f'data/pkl/{file_name}.pkl', 'rb') as f:
        return pickle.load(f)