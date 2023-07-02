import pickle

def save_dictionary(dictionary, name):
    with open(name +'.pkl', 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
    
def load_dictionary(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
