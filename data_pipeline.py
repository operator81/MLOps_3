import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_clean_data(file_path):
    return pd.read_csv(file_path)

def create_pipeline():
    categorical_features = ['Sex', 'Housing', 'Job', 'Purpose']
    numerical_features = ['Age', 'Credit amount', 'Duration']

    # Определение пайплайна для предварительной обработки данных
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Полный пайплайн с моделью
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier())])
    return pipeline

def main():
    data = load_clean_data('cleaned_data.csv')
    X = data.drop('Credit amount', axis=1)  # Здесь укажите вашу целевую переменную
    y = data['Credit amount']  # Убедитесь, что у вас есть целевая переменная

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    print("Модель успешно обучена")

if __name__ == "__main__":
    main()