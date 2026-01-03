import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class PressureDataClassificationPreprocessor1D:
    """Подготовка данных для классификации типа пласта 1D CNN из файлов кривых"""

    def __init__(self, target_size=500, test_size=0.2, val_size=0.1,
                 random_state=42):
        self.target_size = target_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.data_dir = '../dataset/curve/'

        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()


    def load_dataset_for_classification(self):
        """
        Загрузка датасета
        :return: массив DataFrame
        """
        X = []
        y = []

        self._init_target_encoders()

        for item in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, item)

            df = pd.read_csv(file_path)[['t_D', 'P_wD_gauss', 'dP_wD']]

            t_D = np.array(df['t_D'])
            dP_wD = np.array(df['dP_wD'])
            P_wD_gauss = np.array(df['P_wD_gauss'])

            X.append([t_D, dP_wD, P_wD_gauss])

            target = self._encode_new_sample(item.split('_')[0])[0]
            y.append(target)

        return np.array(X), np.array(y)


    def _encode_new_sample(self, new_label):
        """
        Кодирование нового значения (строки)
        :param new_label: строка с меткой
        :return: one-hot закодированный вектор
        """
        integer_encoded = self.label_encoder.transform([new_label])
        integer_encoded_reshaped = integer_encoded.reshape(-1, 1)
        one_hot_encoded = self.one_hot_encoder.transform(integer_encoded_reshaped)
        return one_hot_encoded


    def _init_target_encoders(self):
        """
        Подготовка данных для целевой переменной
        """
        targets = []

        for item in os.listdir(self.data_dir):
            if item.endswith('.csv'):
                label = item.split("_")[0]
                targets.append(label)

        if not targets:
            raise ValueError("В директории не найдено CSV файлов")

        print(f"Найденные метки: {targets}")

        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

        integer_encoded = self.label_encoder.fit_transform(targets)
        integer_encoded = integer_encoded.reshape(-1, 1)

        one_hot_encoded = self.one_hot_encoder.fit_transform(integer_encoded)
        print(f"One-hot кодирование (размерность): {one_hot_encoded.shape}")


if __name__ == "__main__":
    data_preprocess = PressureDataClassificationPreprocessor1D()
    data = data_preprocess.load_dataset_for_classification()
    print(data[0])