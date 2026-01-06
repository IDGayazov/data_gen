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
        self.data_dir = './dataset/curve/'

        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()

        self.debug = debug


    def load_data_for_classification(self):
        """
        Загрузка датасета
        :return: np.array(X), np.array(y)
        """
        X = []
        y = []

        self._init_target_encoders()

        for item in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, item)

            df = pd.read_csv(file_path)[['t_D', 'dP_wD']]

            t_D = np.array(df['t_D'])
            dP_wD = np.array(df['dP_wD'])

            X.append([t_D, dP_wD])

            label = self._get_label_from_filename(item)
            target = self._encode_new_sample(label)[0]
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
                label = self._get_label_from_filename(item)
                targets.append(label)

        if not targets:
            raise ValueError("В директории не найдено CSV файлов")

        if self.debug:
            print(f"Найденные метки: {set(targets)}")

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
            print('X[0] =', X[0])
            print('y[0] =', y[0])

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

        X_train, y_train = clean_nan_data(X_train, y_train, strategy='interpolate')
        X_val, y_val = clean_nan_data(X_val, y_val, strategy='interpolate')
        X_test, y_test = clean_nan_data(X_test, y_test, strategy='interpolate')

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


def check_data_quality(X, y, name="Dataset"):
    """Проверка качества данных"""
    print(f"\n🔍 Проверка данных: {name}")
    print(f"  Форма X: {X.shape}")
    print(f"  Форма y: {y.shape}")

    # Проверка на NaN
    x_nan_count = np.isnan(X).sum()
    y_nan_count = np.isnan(y).sum()
    print(f"  NaN в X: {x_nan_count}")
    print(f"  NaN в y: {y_nan_count}")

    # Проверка на Inf
    x_inf_count = np.isinf(X).sum()
    y_inf_count = np.isinf(y).sum()
    print(f"  Inf в X: {x_inf_count}")
    print(f"  Inf в y: {y_inf_count}")

    # Проверка диапазона значений
    print(f"  X min/max: {X.min():.6f} / {X.max():.6f}")
    print(f"  X mean/std: {X.mean():.6f} / {X.std():.6f}")

    # Проверка меток
    if len(y.shape) > 1:
        print(f"  Распределение меток: {np.sum(y, axis=0)}")
    else:
        unique, counts = np.unique(y, return_counts=True)
        print(f"  Распределение меток: {dict(zip(unique, counts))}")

    return x_nan_count == 0 and y_nan_count == 0 and x_inf_count == 0 and y_inf_count == 0


def clean_nan_data(X, y, strategy='zero'):
    """
    Очистка данных от NaN значений

    Parameters:
    -----------
    X : numpy array
        Входные данные
    y : numpy array
        Метки
    strategy : str
        Стратегия замены NaN:
        - 'zero': замена на 0
        - 'mean': замена на среднее по каналу
        - 'median': замена на медиану по каналу
        - 'interpolate': интерполяция по времени

    Returns:
    --------
    tuple : (X_clean, y_clean)
    """
    X_clean = X.copy()
    y_clean = y.copy()

    nan_count = np.isnan(X_clean).sum()
    print(f"Начальное количество NaN: {nan_count}")

    if nan_count == 0:
        return X_clean, y_clean

    if strategy == 'zero':
        # Замена на 0
        X_clean = np.nan_to_num(X_clean, nan=0.0)
        print(f"✅ NaN заменены на 0")

    elif strategy == 'mean':
        # Замена на среднее по каналу
        for i in range(X_clean.shape[1]):  # По каналам
            channel = X_clean[:, i, :]
            channel_mean = np.nanmean(channel)
            channel_nan_mask = np.isnan(channel)
            channel[channel_nan_mask] = channel_mean
            X_clean[:, i, :] = channel
        print(f"✅ NaN заменены на среднее по каналам")

    elif strategy == 'median':
        # Замена на медиану по каналу
        for i in range(X_clean.shape[1]):
            channel = X_clean[:, i, :]
            channel_median = np.nanmedian(channel)
            channel_nan_mask = np.isnan(channel)
            channel[channel_nan_mask] = channel_median
            X_clean[:, i, :] = channel
        print(f"✅ NaN заменены на медиану по каналам")

    elif strategy == 'interpolate':
        # Линейная интерполяция по времени
        from scipy import interpolate

        for sample_idx in range(X_clean.shape[0]):
            for channel_idx in range(X_clean.shape[1]):
                signal = X_clean[sample_idx, channel_idx, :]

                if np.isnan(signal).any():
                    # Создаем маску валидных точек
                    valid_mask = ~np.isnan(signal)
                    valid_indices = np.where(valid_mask)[0]

                    if len(valid_indices) > 1:
                        # Интерполируем только если есть хотя бы 2 валидные точки
                        f = interpolate.interp1d(
                            valid_indices,
                            signal[valid_mask],
                            kind='linear',
                            fill_value='extrapolate'
                        )
                        all_indices = np.arange(len(signal))
                        signal = f(all_indices)
                        X_clean[sample_idx, channel_idx, :] = signal
                    else:
                        # Если слишком мало валидных точек, заменяем на 0
                        signal[np.isnan(signal)] = 0.0
                        X_clean[sample_idx, channel_idx, :] = signal

        print(f"✅ NaN интерполированы по времени")

    # Удаляем образцы, которые все еще содержат NaN после очистки
    nan_after = np.isnan(X_clean).sum()
    if nan_after > 0:
        print(f"⚠️ После очистки осталось {nan_after} NaN")
        print("Удаляю проблемные образцы...")

        # Находим образцы без NaN
        valid_samples = []
        for i in range(len(X_clean)):
            if not np.isnan(X_clean[i]).any():
                valid_samples.append(i)

        X_clean = X_clean[valid_samples]
        y_clean = y_clean[valid_samples]

        print(f"✅ Оставлено {len(X_clean)} валидных образцов из {len(X)}")

    print(f"✅ Итоговое количество NaN: {np.isnan(X_clean).sum()}")
    return X_clean, y_clean


def analyze_class_separability(X, y, class_names):
    """Анализ различимости классов"""
    print("=" * 80)
    print("АНАЛИЗ РАЗЛИЧИМОСТИ КЛАССОВ")
    print("=" * 80)

    # Преобразуем one-hot в индексы
    if len(y.shape) > 1:
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y

    # 1. Визуализация примеров из каждого класса
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 3, figsize=(15, 20))

    for class_idx, class_name in enumerate(class_names):
        # Находим примеры этого класса
        class_samples = np.where(y_indices == class_idx)[0][:3]  # Первые 3

        for i, sample_idx in enumerate(class_samples):
            ax = axes[class_idx, i]

            # Рисуем оба канала
            time = np.arange(X.shape[2])
            ax.plot(time, X[sample_idx, 0], 'b-', label='Pressure', linewidth=2)
            ax.plot(time, X[sample_idx, 1], 'r-', label='Derivative', linewidth=2)

            ax.set_title(f'{class_name} - Sample {i + 1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Примеры кривых для каждого класса', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('class_examples.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 2. Статистики по классам
    print("\n📊 СТАТИСТИКИ ПО КЛАССАМ:")
    for class_idx, class_name in enumerate(class_names):
        class_mask = y_indices == class_idx
        X_class = X[class_mask]

        print(f"\n{class_name}:")
        for channel in range(X.shape[1]):
            channel_data = X_class[:, channel, :]
            print(f"  Канал {channel}:")
            print(f"    Mean: {channel_data.mean():.4f}, Std: {channel_data.std():.4f}")
            print(f"    Min: {channel_data.min():.4f}, Max: {channel_data.max():.4f}")
            print(f"    Range: {channel_data.max() - channel_data.min():.4f}")

    # 3. Проверка корреляции между классами
    print("\n📈 КОРРЕЛЯЦИЯ МЕЖДУ КЛАССАМИ:")

    # Вычисляем средние кривые для каждого класса
    class_means = []
    for class_idx in range(len(class_names)):
        class_mask = y_indices == class_idx
        class_mean = X[class_mask].mean(axis=0)  # (channels, time)
        class_means.append(class_mean)

    # Вычисляем попарную корреляцию
    from scipy.spatial.distance import cosine

    corr_matrix = np.zeros((len(class_names), len(class_names)))

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            # Flatten и вычисляем косинусное расстояние
            flat_i = class_means[i].flatten()
            flat_j = class_means[j].flatten()
            corr_matrix[i, j] = 1 - cosine(flat_i, flat_j)

    # Визуализация матрицы корреляции
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), class_names)
    plt.title('Косинусная схожесть между средними кривыми классов')
    plt.tight_layout()
    plt.savefig('class_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Выводим самые похожие пары классов
    print("\n🔍 САМЫЕ ПОХОЖИЕ КЛАССЫ:")
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            similarity = corr_matrix[i, j]
            if similarity > 0.95:  # Очень похожи
                print(f"  ⚠️ {class_names[i]} и {class_names[j]}: {similarity:.3f}")
            elif similarity > 0.8:  # Похожи
                print(f"  ⚠️ {class_names[i]} и {class_names[j]}: {similarity:.3f}")

if __name__ == "__main__":
    data_preprocess = PressureDataClassificationPreprocessor1D(debug=False)
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()

    class_names = ['dual_permeability_inf',
                    'radial_composite_inf',
                    'homogeneous_inf',
                    'homogeneous_fin',
                    'dual_porosity_inf',
                    'dual_porosity_fin',
                    'dual_permeability_fin']
    analyze_class_separability(X_train, y_train, class_names)

    # check_data_quality(X_train, y_train)
    #
    # X_train, y_train = clean_nan_data(X_train, y_train, strategy='interpolate')
    # check_data_quality(X_train, y_train)
    #
    # X_val, y_val = clean_nan_data(X_val, y_val, strategy='interpolate')
    # check_data_quality(X_val, y_val)
    #
    # X_test, y_test = clean_nan_data(X_test, y_test, strategy='interpolate')
    # check_data_quality(X_test, y_test)

    data_preprocess.stats()