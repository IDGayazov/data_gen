import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras


class UniversalWellTestTrainer:
    def __init__(self, model, class_names, use_mixed_precision=True):
        """
        Универсальный класс для обучения на GPU и CPU

        Parameters:
        -----------
        model : keras.Model
            Модель для обучения
        class_names : list
            Названия классов
        use_mixed_precision : bool
            Использовать mixed precision для ускорения на GPU
        """
        self.model = model
        self.class_names = class_names

        # Инициализация устройства
        self.device_type = self._setup_device(use_mixed_precision)

        # Пути для сохранения
        self.best_model_path = 'best_welltest_model.h5'
        self.checkpoint_dir = 'training_checkpoints'
        self.metrics_file = 'training_metrics.json'

        # История обучения
        self.history = None
        self.metrics = None
        self.training_time = None

        # Создание директорий
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _setup_device(self, use_mixed_precision=True):
        """
        Настройка устройства (GPU/CPU) и mixed precision

        Returns:
        --------
        str : тип устройства ('GPU' или 'CPU')
        """
        print("=" * 60)
        print("НАСТРОЙКА УСТРОЙСТВА ДЛЯ ОБУЧЕНИЯ")
        print("=" * 60)

        # Проверка доступности GPU
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            try:
                # Включение роста памяти GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Использовать mixed precision на GPU если нужно
                if use_mixed_precision:
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
                    print("✅ Mixed precision включен для GPU")

                # Стратегия распределения (поддержка multi-GPU)
                if len(gpus) > 1:
                    strategy = tf.distribute.MirroredStrategy()
                    print(f"✅ Обнаружено {len(gpus)} GPU. Используется MirroredStrategy")
                    with strategy.scope():
                        # Модель должна быть создана внутри scope
                        pass
                else:
                    strategy = tf.distribute.get_strategy()

                print(f"✅ GPU доступен: {gpus[0].name}")
                print(f"   Всего GPU: {len(gpus)}")
                return 'GPU'

            except Exception as e:
                print(f"⚠️ Ошибка настройки GPU: {e}")
                print("⚠️ Переключаюсь на CPU")
                return 'CPU'
        else:
            print("⚠️ GPU не обнаружен, используется CPU")
            return 'CPU'

    def _create_callbacks(self, X_train=None, batch_size=32, patience=15,
                          reduce_lr_patience=5, monitor='val_accuracy'):
        """
        Создание callback'ов для обучения

        Parameters:
        -----------
        X_train : numpy array, optional
            Тренировочные данные для расчета save_freq
        batch_size : int
            Размер батча
        patience : int
            Терпение для EarlyStopping
        reduce_lr_patience : int
            Терпение для ReduceLROnPlateau
        monitor : str
            Метрика для мониторинга

        Returns:
        --------
        list of keras.callbacks
        """
        # Базовые коллбэки
        callbacks = [
            # Сохранение лучшей модели
            keras.callbacks.ModelCheckpoint(
                filepath=self.best_model_path,
                monitor=monitor,
                mode='max' if 'accuracy' in monitor else 'min',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                save_freq='epoch'
            ),

            # Ранняя остановка
            # keras.callbacks.EarlyStopping(
            #     monitor=monitor,
            #     patience=patience,
            #     restore_best_weights=True,
            #     verbose=1,
            #     mode='max' if 'accuracy' in monitor else 'min'
            # ),

            # Уменьшение learning rate
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1,
                mode='max' if 'accuracy' in monitor else 'min'
            ),

            # CSV логгер
            keras.callbacks.CSVLogger(
                os.path.join(self.checkpoint_dir, 'training_log.csv'),
                separator=',',
                append=True
            ),

            # TerminateOnNaN для безопасности
            keras.callbacks.TerminateOnNaN()
        ]

        # TensorBoard только если есть GPU или специально запрошено
        if self.device_type == 'GPU' or os.getenv('FORCE_TENSORBOARD', '0') == '1':
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.checkpoint_dir, 'tensorboard_logs'),
                    histogram_freq=1 if self.device_type == 'GPU' else 0,
                    write_graph=True,
                    write_images=self.device_type == 'GPU',
                    update_freq='epoch'
                )
            )

        # Backup checkpoints (сохраняем каждые 10 эпох)
        # Используем save_freq вместо period
        if X_train is not None:
            steps_per_epoch = len(X_train) // batch_size
            save_freq = 10 * steps_per_epoch  # Сохранять каждые 10 эпох
        else:
            save_freq = 'epoch'  # Резервный вариант

        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, 'backup_epoch_{epoch:03d}.h5'),
                monitor=monitor,
                save_best_only=False,
                save_weights_only=False,
                verbose=0,
                save_freq=save_freq
            )
        )

        return callbacks

    def _get_optimizer(self, initial_lr=0.01, optimizer_type='adam'):
        """
        Создание оптимизатора в зависимости от устройства

        Parameters:
        -----------
        initial_lr : float
            Начальная скорость обучения
        optimizer_type : str
            Тип оптимизатора ('adam', 'sgd', 'rmsprop')

        Returns:
        --------
        keras.Optimizer
        """
        # Настройки для разных устройств
        if self.device_type == 'GPU':
            clipnorm = 1.0  # Градиентный клиппинг для стабильности
            epsilon = 1e-7  # Для mixed precision
        else:
            clipnorm = 1.0  # Также используем клиппинг на CPU
            epsilon = 1e-7

        if optimizer_type == 'adam':
            optimizer = keras.optimizers.Adam(
                learning_rate=initial_lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=epsilon,
                clipnorm=clipnorm
            )
        elif optimizer_type == 'sgd':
            optimizer = keras.optimizers.SGD(
                learning_rate=initial_lr,
                momentum=0.9,
                nesterov=True,
                clipnorm=clipnorm
            )
        elif optimizer_type == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(
                learning_rate=initial_lr,
                rho=0.9,
                epsilon=epsilon,
                clipnorm=clipnorm
            )
        else:
            raise ValueError(f"Неизвестный оптимизатор: {optimizer_type}")

        return optimizer

    def _compile_model(self, optimizer, loss='categorical_crossentropy',
                       metrics=None, run_eagerly=False):
        """
        Компиляция модели с учетом устройства

        Parameters:
        -----------
        optimizer : keras.Optimizer
            Оптимизатор
        loss : str or keras.Loss
            Функция потерь
        metrics : list
            Список метрик
        run_eagerly : bool
            Запускать ли в eager mode (для отладки)
        """
        if metrics is None:
            metrics = ['accuracy']

        # Настройки компиляции в зависимости от устройства
        compile_kwargs = {
            'optimizer': optimizer,
            'loss': loss,
            'metrics': metrics
        }

        # На GPU можно использовать больший размер батча, на CPU - eager mode для отладки
        if self.device_type == 'CPU' and run_eagerly:
            compile_kwargs['run_eagerly'] = True

        self.model.compile(**compile_kwargs)

    def train(self, X_train, y_train, X_val, y_val,
              batch_size=32, epochs=300, initial_lr=0.01,
              optimizer_type='adam', custom_lr_schedule=None,
              class_weight=None, validation_freq=1):
        """
        Универсальное обучение модели на GPU или CPU
        """
        print("\n" + "=" * 60)
        print(f"НАЧАЛО ОБУЧЕНИЯ НА {self.device_type}")
        print("=" * 60)

        # Автоматическая настройка batch_size для устройства
        adjusted_batch_size = self._adjust_batch_size(batch_size)

        print(f"Устройство: {self.device_type}")
        print(f"Размер батча: {adjusted_batch_size} (запрошен {batch_size})")
        print(f"Тренировочные данные: {X_train.shape}")
        print(f"Валидационные данные: {X_val.shape}")
        print(f"Эпохи: {epochs}")

        # Создание оптимизатора
        optimizer = self._get_optimizer(initial_lr, optimizer_type)

        # Компиляция модели
        self._compile_model(optimizer)

        # Создание расписания learning rate
        if custom_lr_schedule is None:
            # Расписание как в статье
            def lr_schedule(epoch, lr):
                if epoch in [50, 150, 250]:
                    return lr * 0.1
                return lr
        else:
            lr_schedule = custom_lr_schedule

        lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

        # Получение коллбэков
        callbacks = self._create_callbacks()
        callbacks.append(lr_scheduler)

        # Старт таймера
        start_time = time.time()

        try:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=adjusted_batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
                shuffle=True,
                class_weight=class_weight,
                validation_freq=validation_freq
            )

            # Время обучения
            self.training_time = time.time() - start_time

            print(f"\n✅ Обучение завершено за {self.training_time:.2f} секунд")
            print(f"✅ Лучшая модель сохранена в: {self.best_model_path}")

            # Сохранение информации об обучении
            self._save_training_info(X_train, y_train, X_val, y_val)

            return self.history

        except Exception as e:
            print(f"\n❌ Ошибка при обучении: {e}")
            print("Попытка восстановления...")

            # Попытка загрузить лучшую модель
            if os.path.exists(self.best_model_path):
                self.model = keras.models.load_model(self.best_model_path)
                print("✅ Загружена лучшая сохраненная модель")

            raise

    def _adjust_batch_size(self, requested_batch_size):
        """
        Автоматическая корректировка batch_size для устройства

        Parameters:
        -----------
        requested_batch_size : int
            Запрошенный размер батча

        Returns:
        --------
        int : скорректированный размер батча
        """
        if self.device_type == 'GPU':
            # На GPU можно использовать больший batch_size
            gpu_memory = tf.config.list_physical_devices('GPU')[0]
            # Эмпирическая формула для определения максимального batch_size
            max_batch_size = 256  # можно увеличить для мощных GPU
            return min(requested_batch_size, max_batch_size)
        else:
            # На CPU ограничиваем batch_size для экономии памяти
            max_batch_size = 64
            return min(requested_batch_size, max_batch_size)

    def _save_training_info(self, X_train, y_train, X_val, y_val):
        """Сохранение информации об обучении"""
        info = {
            'device_type': self.device_type,
            'training_time': self.training_time,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'input_shape': X_train.shape[1:],
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'model_summary': []
        }

        # Сохранение архитектуры модели
        self.model.summary(print_fn=lambda x: info['model_summary'].append(x))

        # Сохранение в JSON
        with open(self.metrics_file, 'w') as f:
            json.dump(info, f, indent=2, default=str)

        print(f"✅ Информация об обучении сохранена в {self.metrics_file}")

    def evaluate(self, X_test, y_test, batch_size=None):
        """
        Оценка модели на тестовых данных

        Parameters:
        -----------
        X_test, y_test : numpy arrays
            Тестовые данные
        batch_size : int, optional
            Размер батча для оценки

        Returns:
        --------
        dict : метрики оценки
        """
        print("\n" + "=" * 60)
        print("ОЦЕНКА МОДЕЛИ")
        print("=" * 60)

        # Автоматический выбор batch_size для оценки
        if batch_size is None:
            batch_size = self._adjust_batch_size(64)  # Меньше batch_size для оценки

        # Базовая оценка
        test_loss, test_accuracy = self.model.evaluate(
            X_test, y_test,
            batch_size=batch_size,
            verbose=0
        )

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Предсказания
        y_pred = self.model.predict(X_test, batch_size=batch_size, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Детальная оценка
        from sklearn.metrics import (classification_report, confusion_matrix,
                                     precision_recall_fscore_support, roc_auc_score)

        try:
            # ROC AUC (только для бинарной или многоклассовой с one-vs-rest)
            if len(self.class_names) == 2:
                roc_auc = roc_auc_score(y_true, y_pred[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
        except:
            roc_auc = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)

        # Classification report
        report = classification_report(
            y_true, y_pred_classes,
            target_names=self.class_names,
            digits=4,
            output_dict=True
        )

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred_classes, average=None
        )

        # Вывод результатов
        print("\n📊 Classification Report:")
        print(classification_report(
            y_true, y_pred_classes,
            target_names=self.class_names,
            digits=4
        ))

        print(f"\n📈 ROC AUC Score: {roc_auc:.4f}" if roc_auc is not None else "")
        print(f"\n📊 Confusion Matrix:")
        print(cm)

        # Сохранение метрик
        self.metrics = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'y_true': y_true.tolist(),
            'y_pred': y_pred_classes.tolist(),
            'y_pred_proba': y_pred.tolist(),
            'device_used': self.device_type,
            'evaluation_time': time.time()
        }

        # Сохранение метрик в файл
        with open('evaluation_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        return self.metrics

    def predict_batch(self, X, batch_size=None):
        """
        Предсказание для батча данных

        Parameters:
        -----------
        X : numpy array
            Данные для предсказания
        batch_size : int, optional
            Размер батча

        Returns:
        --------
        tuple : (predicted_classes, probabilities)
        """
        if batch_size is None:
            batch_size = self._adjust_batch_size(32)

        probabilities = self.model.predict(X, batch_size=batch_size, verbose=0)
        predicted_classes = np.argmax(probabilities, axis=1)

        return predicted_classes, probabilities

    def predict_single(self, X_single):
        """
        Предсказание для одного примера

        Parameters:
        -----------
        X_single : numpy array
            Один пример данных

        Returns:
        --------
        tuple : (predicted_class, probabilities, confidence)
        """
        X_batch = np.expand_dims(X_single, axis=0)

        # Используем predict с batch_size=1 для стабильности
        probabilities = self.model.predict(X_batch, batch_size=1, verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]

        return predicted_class, probabilities, confidence

    def benchmark_inference(self, X_sample, n_iterations=100):
        """
        Бенчмарк скорости инференса

        Parameters:
        -----------
        X_sample : numpy array
            Пример данных для тестирования
        n_iterations : int
            Количество итераций для бенчмарка

        Returns:
        --------
        dict : результаты бенчмарка
        """
        print(f"\n🏃 Бенчмарк инференса на {self.device_type}...")

        # Подготовка данных
        X_test = np.repeat(X_sample[np.newaxis, :, :], 100, axis=0)

        # Разогрев
        _ = self.model.predict(X_test[:1], verbose=0)

        # Замер времени
        start_time = time.time()

        for _ in range(n_iterations):
            _ = self.model.predict(X_test, batch_size=32, verbose=0)

        total_time = time.time() - start_time
        avg_time_per_batch = total_time / n_iterations
        avg_time_per_sample = avg_time_per_batch / len(X_test)

        results = {
            'device': self.device_type,
            'total_time_seconds': total_time,
            'iterations': n_iterations,
            'batch_size': len(X_test),
            'avg_time_per_batch_ms': avg_time_per_batch * 1000,
            'avg_time_per_sample_ms': avg_time_per_sample * 1000,
            'samples_per_second': 1 / avg_time_per_sample
        }

        print(f"\n📊 Результаты бенчмарка:")
        print(f"  Устройство: {results['device']}")
        print(f"  Среднее время на образец: {results['avg_time_per_sample_ms']:.2f} ms")
        print(f"  Образцов в секунду: {results['samples_per_second']:.0f}")

        return results

    def plot_training_history(self, save_path=None):
        """Визуализация истории обучения"""
        if self.history is None:
            print("⚠️ Модель еще не обучена!")
            return

        import matplotlib.pyplot as plt

        history_dict = self.history.history

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history_dict['accuracy'], label='Train', linewidth=2)
        axes[0].plot(history_dict['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title(f'Model Accuracy ({self.device_type})', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history_dict['loss'], label='Train', linewidth=2)
        axes[1].plot(history_dict['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title(f'Model Loss ({self.device_type})', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 График сохранен в {save_path}")

        plt.show()

    def save_model(self, path='welltest_model_complete', format='keras'):
        """
        Сохранение модели в современных форматах Keras 3.0

        Parameters:
        -----------
        path : str
            Базовое имя файла
        format : str
            Формат сохранения ('keras', 'h5', 'savedmodel', 'all')
        """
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        if format in ['h5', 'all']:
            # Для H5 формата в Keras 3.0 нужно явно указать .h5
            h5_path = f"{path}.h5" if not path.endswith('.h5') else path
            try:
                # В Keras 3.0 save использует .keras по умолчанию
                # Для сохранения в H5 нужно использовать save_weights с правильным расширением
                self.model.save_weights(f"{path}.weights.h5")
                print(f"✅ Веса модели сохранены в H5: {path}.weights.h5")
            except Exception as e:
                print(f"⚠️ Не удалось сохранить в H5: {e}")

        if format in ['keras', 'all']:
            # Рекомендуемый формат Keras 3.0
            keras_path = f"{path}.keras" if not path.endswith('.keras') else path
            try:
                self.model.save(keras_path)
                print(f"✅ Модель сохранена в Keras format: {keras_path}")
            except Exception as e:
                print(f"⚠️ Не удалось сохранить в Keras format: {e}")

        if format in ['savedmodel', 'all']:
            # SavedModel формат
            savedmodel_path = f"{path}_savedmodel"
            try:
                tf.saved_model.save(self.model, savedmodel_path)
                print(f"✅ Модель сохранена в SavedModel: {savedmodel_path}")
            except Exception as e:
                print(f"⚠️ Не удалось сохранить в SavedModel: {e}")

        # Сохранение конфигурации
        config = {
            'class_names': self.class_names,
            'input_shape': self.model.input_shape[1:],
            'device_used': self.device_type,
            'model_format': format,
            'save_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        config_path = f"{path}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Конфигурация сохранена: {config_path}")

    def load_model(self, path, custom_objects=None):
        """
        Загрузка модели из различных форматов

        Parameters:
        -----------
        path : str
            Путь к модели (может быть .keras, .h5, или директория SavedModel)
        custom_objects : dict
            Пользовательские объекты для загрузки
        """
        if custom_objects is None:
            custom_objects = {}

        print(f"\n📥 Загрузка модели из: {path}")

        # Определяем формат по расширению
        if path.endswith('.keras'):
            # Keras 3.0 формат
            self.model = keras.models.load_model(path, custom_objects=custom_objects)
            print(f"✅ Модель загружена из .keras формата")

        elif path.endswith('.h5'):
            # HDF5 формат
            try:
                self.model = keras.models.load_model(path, custom_objects=custom_objects)
                print(f"✅ Модель загружена из .h5 формата")
            except:
                # Может быть это только веса
                print(f"⚠️ Попытка загрузить только веса...")
                # Нужно сначала создать модель, потом загрузить веса
                # Это зависит от вашей архитектуры

        elif os.path.isdir(path) and 'saved_model.pb' in os.listdir(path):
            # SavedModel формат
            self.model = tf.saved_model.load(path)
            print(f"✅ Модель загружена из SavedModel формата")

        elif path.endswith('.weights.h5'):
            # Только веса
            print(f"⚠️ Загружены только веса. Нужна архитектура модели.")
            # Здесь нужно создать модель с той же архитектурой
            # self.model = create_model_with_same_architecture()
            # self.model.load_weights(path)

        else:
            raise ValueError(f"Неизвестный формат модели: {path}")

        # Загрузка метаданных если есть
        metadata_path = path.replace('.keras', '_metadata.json').replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.class_names = metadata.get('class_names', self.class_names)
            print(f"✅ Метаданные загружены из {metadata_path}")

        return self.model