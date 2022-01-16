import pandas as pd

def feature_selection(x, y):
    pass

def train_model(model, x, y):
    model.fit(x, y)


def main():
    data_path = f'data/Genes/Original'
    print('reading data...')
    x = pd.read_csv(f'{data_path}/data.csv').set_index('Unnamed: 0')
    y = pd.read_csv(f'{data_path}/labels.csv').set_index('Unnamed: 0').Class
    pass

if __name__ == '__main__':
    main()
