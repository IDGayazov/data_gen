import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class PressureDataClassificationPreprocessor1D:
    """Подготовка данных для классификации типа пласта 1D CNN из файлов кривых"""

    def __init__(self, target_size=500, test_size=0.2, val_size=0.1,
                 random_state=42, debug=False):
        self.target_size = target_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.data_dir = './dataset4/curve/'

        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()

        self.class_names = []

        self.debug = debug


    def load_data_for_classification(self):
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


    def normalize(self, X):
        """
        Принимает:
        -----------
        X : numpy array, shape (samples, timesteps, channels)
            Входные данные, где:
            X[:, :, 0] = p_D (безразмерная производная давления)
            X[:, :, 1] = t_D (безразмерное время)

        Возвращает:
        --------
        X_norm : numpy array, shape (samples, timesteps, channels)
            Нормализованные данные
        """
        X_norm = X.copy()

        for i in range(X.shape[2]):
            channel = X[:, :, i]

            channel_min = np.min(channel, axis=1, keepdims=True)
            channel_max = np.max(channel, axis=1, keepdims=True)
            channel_range = channel_max - channel_min

            channel_norm = (channel - channel_min) / channel_range

            X_norm[:, :, i] = channel_norm

        return X_norm


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
                label = self._get_label_from_filename(item)
                targets.append(label)

        for label in set(targets):
            self.class_names.append(label)

        if not self.class_names:
            raise ValueError("В директории не найдено CSV файлов")

        print(f"Найденные метки: {self.class_names}")

        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

        integer_encoded = self.label_encoder.fit_transform(targets)
        integer_encoded = integer_encoded.reshape(-1, 1)

        one_hot_encoded = self.one_hot_encoder.fit_transform(integer_encoded)

        if self.debug:
            print(f"One-hot кодирование (размерность): {one_hot_encoded.shape}")


    def _get_label_from_filename(self, filename):
        parts = filename.rsplit('_', 1)
        label = parts[0]
        return label


    def get_dataset(self):
        X, y = self.load_data_for_classification()

        if self.debug:
            print('shape X =', X.shape)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=42,
            shuffle=True,
            stratify=y
        )

        val_size_relative = self.val_size / (1 - self.test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size_relative,
            random_state=42,
            shuffle=True,
            stratify=y_train_val
        )

        if self.debug:
            print(f"Train: {X_train.shape} ({(X_train.shape[0] / X.shape[0]) * 100:.1f}%)")
            print(f"Val: {X_val.shape} ({(X_val.shape[0] / X.shape[0]) * 100:.1f}%)")
            print(f"Test: {X_test.shape} ({(X_test.shape[0] / X.shape[0]) * 100:.1f}%)")

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.X_original, self.y_original = X, y

        return X_train, X_val, X_test, y_train, y_val, y_test


    def stats(self, return_dict=False):
        """
        Выводит статистику распределения меток по подвыборкам.
        Работает с one-hot encoded метками.
        """
        if not hasattr(self, 'y_original'):
            raise ValueError("Сначала вызовите get_dataset()")

        print(f"Формат y_original: {self.y_original.shape}")
        print(f"Пример y_original[0]: {self.y_original[0]}")

        if len(self.y_original.shape) > 1 and self.y_original.shape[1] > 1:
            print("Обнаружены one-hot encoded метки. Преобразую в целые числа...")
            y_original_int = np.argmax(self.y_original, axis=1)
            y_train_int = np.argmax(self.y_train, axis=1)
            y_val_int = np.argmax(self.y_val, axis=1)
            y_test_int = np.argmax(self.y_test, axis=1)
        else:
            y_original_int = self.y_original.flatten()
            y_train_int = self.y_train.flatten()
            y_val_int = self.y_val.flatten()
            y_test_int = self.y_test.flatten()

        datasets = {
            'Original': y_original_int,
            'Train': y_train_int,
            'Validation': y_val_int,
            'Test': y_test_int
        }

        stats_dict = {}

        print("=" * 70)
        print("СТАТИСТИКА РАСПРЕДЕЛЕНИЯ МЕТОК")
        print("=" * 70)

        for name, y_data in datasets.items():
            total = len(y_data)
            counter = Counter(y_data)

            sorted_classes = sorted(counter.items())

            stats = {
                'total': total,
                'counts': dict(sorted_classes),
                'percentages': {k: v / total * 100 for k, v in sorted_classes}
            }

            stats_dict[name] = stats

            print(f"\n{name} (n={total}):")
            print("-" * 40)

            for cls, count in sorted_classes:
                percentage = count / total * 100
                print(f"  Класс {cls}: {count:4d} ({percentage:6.2f}%)")

        print("\n" + "=" * 70)
        print("СВОДНАЯ ТАБЛИЦА")
        print("=" * 70)

        all_classes = sorted(set(y_original_int))

        header = f"{'Класс':<8} {'Original':<12} {'Train':<12} {'Val':<12} {'Test':<12}"
        print(header)
        print("-" * 60)

        for cls in all_classes:
            row = f"{cls:<8}"
            for name in ['Original', 'Train', 'Validation', 'Test']:
                count = stats_dict[name]['counts'].get(cls, 0)
                total = stats_dict[name]['total']
                percentage = count / total * 100 if total > 0 else 0
                row += f"{count} ({percentage:5.1f}%)  "
            print(row)

        if return_dict:
            return stats_dict


    def get_class_names(self):
        return self.class_names


if __name__ == "__main__":
    # count_omega = 0
    # count_lambda = 0
    #
    # for item in os.listdir('./dataset/params'):
    #     file_path = os.path.join('./dataset/params', item)
    #     df = pd.read_csv(file_path)
    #
    #     count_omega += df['omega'].between(0.01, 0.5).sum()
    #     count_lambda += df['lambda'].between(1e-8, 1e-5).sum()
    #
    # print('omega cnt: ', count_omega)
    # print('lambda cnt: ', count_lambda)

    data_preprocess = PressureDataClassificationPreprocessor1D(debug=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()

    # data_preprocess.stats()