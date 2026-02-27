import os
from typing import override

import numpy as np
import pandas as pd

from preprocess.abstract_preprcessor import AbstractPressureDataClassificationPreprocessor


class PressureDataClassificationPreprocessor2D(AbstractPressureDataClassificationPreprocessor):
    """
    Загрузка датасета для классификации через для обучения модели 2D CNN.
    Данные формируются в виде time_series_size x time_series_size.
    [кривая давления^T * (time_series_size / 2) | кривая производной давления^T * (time_series_size / 2)].
    """
    def __init__(self, data_dir, test_size=0.2, val_size=0.1, random_state=42, debug=False):
        super().__init__(data_dir, test_size, val_size, random_state, debug)

    @override
    def load_data(self):
        """
        Загрузка датасета и преобразование в матрицы 128×128 согласно статье
        :return: np.array(X), np.array(y)
        """
        X_list = []
        y_list = []

        self._init_target_encoders()

        for item in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, item)

            df = pd.read_csv(file_path)[['t_D', 'dP_wD', 'P_wD']]

            # Извлекаем данные
            t_D = np.array(df['t_D'])            # время (128 точек)
            dP_wD = np.array(df['dP_wD'])        # производная давления (128 точек)
            P_wD = np.array(df['P_wD'])          # изменение давления (128 точек)

            # Проверяем, что у нас 128 точек
            assert len(dP_wD) == 128, f"Ожидается 128 точек давления, получено {len(dP_wD)}"
            assert len(P_wD) == 128, f"Ожидается 128 точек производной, получено {len(P_wD)}"

            # Дублируем каждый вектор 64 раз для создания матрицы 128×64
            # pressure_matrix = P_wD.reshape(128, 1).repeat(64, axis=1)
            # derivative_matrix = dP_wD.reshape(128, 1).repeat(64, axis=1)
            pressure_matrix = np.tile(P_wD.reshape(-1, 1), (1, 64))
            derivative_matrix = np.tile(dP_wD.reshape(-1, 1), (1, 64))

            # Склеиваем матрицы горизонтально: слева давление, справа производная
            # Получаем матрицу 128×128
            sample_matrix = np.hstack([pressure_matrix, derivative_matrix])

            # Добавляем размерность канала для CNN (batch_size, height, width, channels)
            sample_matrix = sample_matrix.reshape(128, 128, 1)

            X_list.append(sample_matrix)

            # Получаем метку как в исходном коде
            label = self._get_label_from_filename(item)
            target = self._encode_new_sample(label)[0]
            y_list.append(target)

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"Загружено {len(X)} образцов")
        print(f"Форма X: {X.shape}")  # (n_samples, 128, 128, 1)
        print(f"Форма y: {y.shape}")

        return X, y

if __name__ == "__main__":
    data_preprocess = PressureDataClassificationPreprocessor2D(data_dir='../datasets/dataset/curve', debug=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()

    data_preprocess.stats()

    X_train_norm = data_preprocess.normalize(X_train)

    df = pd.DataFrame(X_train_norm[0][:, :, 0])
    print(df)