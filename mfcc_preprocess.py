# загрузка данных через выделение важных признаков (n_mfcc)

import os
import librosa
import numpy as np

def load_and_preprocess_data(directory, sample_rate=22050*2, duration=8.0, n_mfcc=20):
    X = []
    labels = []
    max_slices = 0  # Переменная для хранения максимального количества временных срезов

    for label_idx, label_name in enumerate(os.listdir(directory)):
        label_dir = os.path.join(directory, label_name)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                file_path = os.path.join(label_dir, filename)
                try:
                    # Загрузка аудиофайла и преобразование в MFCC
                    y, sr = librosa.load(file_path, sr=sample_rate, duration=duration, mono=True)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

                    # Обновление максимального количества временных срезов
                    max_slices = max(max_slices, mfccs.shape[1])

                    # Добавление MFCC в список признаков и меток классов
                    X.append(mfccs)
                    labels.append(label_idx)
                except Exception as e:
                    print(f"Ошибка при обработке аудиофайла {file_path}: {e}")

    # Выравнивание массивов MFCC по максимальному количеству временных срезов
    X = align_mfccs(X, max_slices)

    return np.array(X), np.array(labels)

def align_mfccs(mfccs_list, max_slices):
    aligned_mfccs_list = []
    for mfccs in mfccs_list:
        # Вычисляем количество временных срезов, которые нужно добавить
        padding_slices = max_slices - mfccs.shape[1]
        # Добавляем нулевые паддинги к массиву MFCC
        padded_mfccs = np.pad(mfccs, ((0, 0), (0, padding_slices)), mode='constant')
        aligned_mfccs_list.append(padded_mfccs)
    return aligned_mfccs_list


# Путь к папке с аудиофайлами
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
data_directory = '/content/drive/MyDrive/noisy_dataset'

# Загрузка данных и их предварительная обработка
X, y = load_and_preprocess_data(data_directory)

# Преобразование меток в формат one-hot
from sklearn.preprocessing import LabelEncoder

# Преобразование строковых меток классов в уникальные целочисленные идентификаторы
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Преобразование уникальных идентификаторов в one-hot представление
y_categorical = to_categorical(y_encoded)

from sklearn.model_selection import train_test_split
# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.2)
# Распечатка размерности данных
print("Train images shape",X_train.shape, y_train.shape)
print("Test images shape",X_test.shape, y_test.shape)
print("Evaluate image shape",X_val.shape, y_val.shape)