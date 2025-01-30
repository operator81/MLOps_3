import pandas as pd

def load_data(file_path):
    # Загрузка данных из csv файла
    data = pd.read_csv(file_path)
    return data

def prepare_data(data):
    # Заполнение пропусков
    data.fillna({'Saving accounts': 'unknown', 'Checking account': 'unknown'}, inplace=True)
    
    # Преобразование категориальных признаков в числовые
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Housing'] = data['Housing'].map({'own': 0, 'rent': 1, 'free': 2})
    data['Job'] = data['Job'].astype('category').cat.codes  # Преобразуем в категории
    data['Purpose'] = data['Purpose'].astype('category').cat.codes  # То же самое для цели

    return data

if __name__ == "__main__":
    file_path = 'data.csv'  
    data = load_data(file_path)
    prepared_data = prepare_data(data)
    
    # Сохраните подготовленные данные
    prepared_data.to_csv('prepared_data.csv', index=False)