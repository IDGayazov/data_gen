import keras
import numpy as np
from keras import layers

from train.data_preprocess_1d import PressureDataClassificationPreprocessor1D
from train.main import simple_normalization, fix_data_shape


# def fix_data_shape(X):
#     """Исправление shape данных для Conv1D"""
#     # Из (samples, channels, timesteps) в (samples, timesteps, channels)
#     if X.shape[1] == 2 and X.shape[2] == 128:  # текущий неправильный формат
#         print(f"Исправление shape: {X.shape} -> ", end="")
#         X_fixed = np.transpose(X, (0, 2, 1))  # меняем оси 1 и 2 местами
#         print(f"{X_fixed.shape}")
#         return X_fixed
#     return X


# ИСПРАВЛЕННЫЙ диагностический тест
def run_diagnostic_test_fixed(X_train, y_train):
    """Быстрый диагностический тест с исправленным shape"""

    print("=== ДИАГНОСТИЧЕСКИЙ ТЕСТ (ИСПРАВЛЕННЫЙ) ===")

    # 1. ИСПРАВЛЯЕМ SHAPE!
    print(f"Исходный X_train shape: {X_train.shape}")
    X_train_fixed = fix_data_shape(X_train)
    print(f"Исправленный X_train shape: {X_train_fixed.shape}")

    # 2. Проверяем формат y_train
    print(f"\ny_train shape: {y_train.shape}")

    # Преобразуем one-hot в binary если нужно
    if len(y_train.shape) == 2 and y_train.shape[1] == 2:
        y_train_binary = np.argmax(y_train, axis=1)
    else:
        y_train_binary = y_train

    print(f"y_train_binary unique: {np.unique(y_train_binary)}")
    print(f"Class distribution: 0={sum(y_train_binary == 0)}, 1={sum(y_train_binary == 1)}")

    # 3. Простейшая модель (теперь с правильным input_shape!)
    test_model = keras.Sequential([
        layers.Input(shape=X_train_fixed.shape[1:]),  # (128, 2)
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])

    test_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 4. Проверка на 10 образцах
    n_test = 10
    X_small = X_train_fixed[:n_test]
    y_small = y_train_binary[:n_test]

    print(f"\nТест на запоминание {n_test} образцов:")
    print(f"  X_small shape: {X_small.shape}")
    print(f"  y_small values: {y_small}")

    # Обучаем с маленьким learning rate
    test_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # МАЛЕНЬКИЙ LR!
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = test_model.fit(
        X_small, y_small,
        epochs=50,
        verbose=0
    )

    preds = test_model.predict(X_small, verbose=0)
    accuracy = np.mean((preds > 0.5).flatten() == y_small)
    print(f"  Final accuracy: {accuracy:.2%}")
    print(f"  Final loss: {history.history['loss'][-1]:.4f}")

    # 5. Проверка предсказаний
    print(f"\nПримеры предсказаний:")
    for i in range(min(5, n_test)):
        print(f"  Sample {i}: y={y_small[i]}, pred={preds[i][0]:.3f}")

    return X_train_fixed, y_train_binary


def check_data_statistics(X, y, name="Data"):
    """Детальная проверка статистики данных"""
    print(f"\n=== СТАТИСТИКА {name} ===")
    print(f"Shape: {X.shape}")

    # Проверка значений
    print(f"\nОбщая статистика:")
    print(f"  Min: {X.min():.6e}")
    print(f"  Max: {X.max():.6e}")
    print(f"  Mean: {X.mean():.6e}")
    print(f"  Std: {X.std():.6e}")

    # Проверка по каналам
    print(f"\nПо каналам:")
    for i in range(X.shape[2]):
        channel_data = X[:, :, i]
        print(f"  Канал {i}: min={channel_data.min():.6e}, max={channel_data.max():.6e}, "
              f"mean={channel_data.mean():.6e}, std={channel_data.std():.6e}")

    # Проверка NaN/Inf
    print(f"\nПроверка корректности:")
    print(f"  NaN values: {np.isnan(X).sum()}")
    print(f"  Inf values: {np.isinf(X).sum()}")
    print(f"  Zero values: {(X == 0).sum()} / {X.size} ({100 * (X == 0).sum() / X.size:.1f}%)")

    # Проверка меток
    print(f"\nМетки (y):")
    print(f"  Unique values: {np.unique(y)}")
    if len(y.shape) == 2:
        print(f"  One-hot, Class 0: {sum(y[:, 0] == 1)}, Class 1: {sum(y[:, 1] == 1)}")
    else:
        print(f"  Binary, Class 0: {sum(y == 0)}, Class 1: {sum(y == 1)}")


if __name__ == "__main__":
    data_preprocess = PressureDataClassificationPreprocessor1D(debug=False)
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()

    # run_diagnostic_test_fixed(X_train, y_train)
    X_train = fix_data_shape(X_train)
    # X_train = simple_normalization(X_train)

    # Проверьте:
    # - Правильность меток
    # - Нормализацию
    # - Разделимость классов
    print("Статистика по классам:")
    print(f"Класс 0 (homogeneous_inf): {np.sum(y_train[:, 0])} образцов")
    print(f"Класс 1 (dual_porosity_inf): {np.sum(y_train[:, 1])} образцов")

    # Проверьте, действительно ли данные различаются
    mean_class0 = X_train[y_train[:, 0] == 1].mean(axis=(0, 1))
    mean_class1 = X_train[y_train[:, 1] == 1].mean(axis=(0, 1))
    print(f"\nСредние значения по классам:")
    print(f"Класс 0: {mean_class0}")
    print(f"Класс 1: {mean_class1}")

    # check_data_statistics(X_train, y_train)