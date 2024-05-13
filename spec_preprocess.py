# третий
import os
import librosa
import numpy as np
import cv2
from tqdm import tqdm

def load_and_preprocess_data(data_directory, img_size=(100, 100)):
    X = []
    y = []

    # Сканируем директорию для файлов
    for label in os.listdir(data_directory):
        label_dir = os.path.join(data_directory, label)
        if os.path.isdir(label_dir):
            for file in tqdm(os.listdir(label_dir), desc=f"Processing {label}"):
                file_path = os.path.join(label_dir, file)
                # Загружаем аудиозапись с использованием librosa
                try:
                    audio, sr = librosa.load(file_path, sr=None, mono=True)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

                # Преобразуем аудио в мел-спектрограмму
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # Приводим к диапазону [0, 255] для корректного отображения
                norm_spec = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX)

                # Масштабируем спектрограмму до размера 100x100
                resized_spec = cv2.resize(norm_spec, img_size)

                # Преобразуем в формат с 3 каналами (RGB)
                image_3ch = cv2.cvtColor(resized_spec, cv2.COLOR_GRAY2RGB)

                # Добавляем в датасет
                X.append(image_3ch)
                y.append(label)

    # Преобразуем в numpy массивы для дальнейшей обработки
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y

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
