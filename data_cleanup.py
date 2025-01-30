import pandas as pd

def load_data(file_path):
    # Загрузка подготовленных данных
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Удаление дубликатов
    data.drop_duplicates(inplace=True)

    # Удаление строк с большим количеством NaN
    data.dropna(thresh=len(data.columns)-3, inplace=True)

    return data

if __name__ == "__main__":
    file_path = 'prepared_data.csv'  
    data = load_data(file_path)
    cleaned_data = clean_data(data)
    
    # сохраняем очищенные данные
    cleaned_data.to_csv('cleaned_data.csv', index=False)