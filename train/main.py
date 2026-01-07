import numpy as np

from train.data_preprocess_1d import PressureDataClassificationPreprocessor1D
from train.model import WellTest1DCNN
from train.train import UniversalWellTestTrainer


def normalize(X):
    """
    Parameters:
    -----------
    X : numpy array, shape (samples, timesteps, channels)
        Входные данные, где:
        X[:, :, 0] = p_D (безразмерная производная давления)
        X[:, :, 1] = t_D (безразмерное время)

    Returns:
    --------
    X_norm : numpy array, shape (samples, timesteps, channels)
        Нормализованные данные
    """
    X_norm = X.copy()

    for i in range(X.shape[2]):  # По каналам (0 и 1)
        # Берем данные канала
        channel = X[:, :, i]

        # Нормализация каждого образца отдельно
        # p_min = np.min(p_Dk, axis=1, keepdims=True) для всего датасета сразу
        channel_min = np.min(channel, axis=1, keepdims=True)  # Минимум для каждого образца
        channel_max = np.max(channel, axis=1, keepdims=True)  # Максимум для каждого образца
        channel_range = channel_max - channel_min

        # Избегаем деления на 0
        # Создаем маску где range очень маленький
        small_range_mask = channel_range < 1e-10

        # Нормализуем: (x - min) / (max - min)
        channel_norm = (channel - channel_min) / channel_range

        # Для образцов с очень маленьким range, просто вычитаем min
        if np.any(small_range_mask):
            # Находим индексы где нужно исправить
            sample_indices = np.where(small_range_mask.flatten())[0]
            for sample_idx in sample_indices:
                channel_norm[sample_idx, :] = channel[sample_idx, :] - channel_min[sample_idx, 0]

        X_norm[:, :, i] = channel_norm

    return X_norm


if __name__ == "__main__":
    model = WellTest1DCNN(num_classes=2, input_shape=(128, 2))
    model.compile_model(learning_rate=0.01)

    trainer = UniversalWellTestTrainer(model.model)

    data_preprocess = PressureDataClassificationPreprocessor1D(debug=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()

    X_train = normalize(X_train)
    X_val = normalize(X_val)
    X_test = normalize(X_test)

    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        epochs=10,
        initial_lr=0.00001
    )

    metrics = trainer.evaluate(X_test, y_test)

    sample = X_test[0]
    benchmark = trainer.benchmark_inference(sample, n_iterations=10)

    trainer.plot_training_history(save_path='training_results.png')

    trainer.save_model('my_universal_model', format='both')

    pred_class, probs, confidence = trainer.predict_single(X_test[0])
    print(f"Класс: {pred_class}, Уверенность: {confidence:.2%}")