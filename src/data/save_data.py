import os
import json
import pickle

def save_json_data(data, file_name):
    # Get the root directory of your project
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Construct the full path to the directory
    dir_path = os.path.join(root_dir, 'src/data/json')

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Construct the full path to the file
    file_path = os.path.join(dir_path, f'{file_name}.json')

    print(f'Saving data to {file_path}')

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)

def save_pickled_data(data, file_name):
    # Get the root directory of your project
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Construct the full path to the directory
    dir_path = os.path.join(root_dir, 'src/data/pkl')

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Construct the full path to the file
    file_path = os.path.join(dir_path, f'{file_name}.pkl')

    print(f'Saving data to {file_path}')

    with open(file_path, 'wb') as file:
        pickle.dump(data, file)