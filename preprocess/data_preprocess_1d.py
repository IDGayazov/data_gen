import os

import numpy as np
import pandas as pd
from typing_extensions import override

from preprocess.abstract_preprcessor import AbstractPressureDataClassificationPreprocessor


class PressureDataClassificationPreprocessor1D(AbstractPressureDataClassificationPreprocessor):
    """Подготовка данных для классификации типа пласта из файлов кривых"""

    def __init__(self, data_dir, test_size=0.2, val_size=0.1, random_state=42, debug=False):
        super().__init__(data_dir, test_size, val_size, random_state, debug)

    @override
    def load_data(self):
        """
        Загрузка датасета
        :return: np.array(X), np.array(y)
        """
        X_list = []
        y_list = []

        self._init_target_encoders()

        for item in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, item)

            df = pd.read_csv(file_path)[['t_D', 'dP_wD']]

            t_D = np.array(df['t_D'])
            dP_wD = np.array(df['dP_wD'])

            sample = np.column_stack([dP_wD, t_D])
            X_list.append(sample)

            label = self._get_label_from_filename(item)
            target = self._encode_new_sample(label)[0]
            y_list.append(target)

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y


if __name__ == "__main__":
    data_preprocess = PressureDataClassificationPreprocessor1D(data_dir='../datasets/dataset/curve', debug=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()

    data_preprocess.stats()