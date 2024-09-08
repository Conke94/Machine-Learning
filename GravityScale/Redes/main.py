import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def cross_validation(data, labels, k, epochs, batch_size):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model = create_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        _, accuracy = model.evaluate(X_test, y_test, verbose=0)

        accuracies.append(accuracy)
        print(f'Acurácia da partição: {accuracy * 100:.2f}%')

    return np.mean(accuracies)

def create_model(input):
    model = Sequential()

    model.add(Dense(16, input_dim=input, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(labels.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

data = pd.read_csv('dados_treinamento.csv')

features = data[['qualidade_pressao_arterial', 'pulso', 'respiracao']].values
labels = data['rotulo'].values

scaler = StandardScaler()
features = scaler.fit_transform(features)

labels = to_categorical(labels) # Codificação one hot

average_accuracy = cross_validation(features, labels, k=4, epochs=42, batch_size=6)
print(f'Acurácia média: {average_accuracy * 100:.2f}%')