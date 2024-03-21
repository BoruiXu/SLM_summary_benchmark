import datasets

#xsum cnndm newsroom
def load_data(path):
    data = datasets.load_dataset('json',data_files=path)['train']
    return data

