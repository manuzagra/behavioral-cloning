import pickle


def save_data(data, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(e)
        return False

def load_data(path):
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return d
    except Exception as e:
        print(e)
        return None