import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score)


class WellTestTrainer:
    TASK_TYPES = ['classification', 'regression']

    def __init__(self, model, task_type='classification', class_names=None,
                 output_names=None, use_mixed_precision=True):
        """
        Класс для обучения моделей (поддерживает классификацию и регрессию)

        Parameters:
        -----------
        model : keras.Model
            Модель для обучения
        task_type : str
            Тип задачи ('classification' или 'regression')
        class_names : list, optional
            Названия классов (для классификации)
        output_names : list, optional
            Названия выходных параметров (для регрессии)
        use_mixed_precision : bool
            Использовать mixed precision для ускорения на GPU
        """
        if task_type not in self.TASK_TYPES:
            raise ValueError(f"task_type должен быть одним из {self.TASK_TYPES}")

        self.model = model
        self.task_type = task_type
        self.class_names = class_names or []
        self.output_names = output_names or [f'output_{i}' for i in range(model.output_shape[-1])]

        self.num_outputs = model.output_shape[-1]

        self.device_type = self._setup_device(use_mixed_precision)

        self.best_model_path = f'best_model_{task_type}.h5'
        self.checkpoint_dir = f'training_checkpoints_{task_type}'
        self.metrics_file = f'training_metrics_{task_type}.json'

        self.history = None
        self.metrics = None
        self.training_time = None
        self.scaler_y = None

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _setup_device(self, use_mixed_precision=True):
        """Настройка устройства (GPU/CPU)"""
        print("=" * 60)
        print(f"НАСТРОЙКА УСТРОЙСТВА ДЛЯ ОБУЧЕНИЯ ({self.task_type.upper()})")
        print("=" * 60)

        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            print(f"Обнаружено GPU: {len(gpus)}")
            if use_mixed_precision:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("✅ Mixed precision включен")
            return 'GPU'
        else:
            print("GPU не обнаружен, используется CPU")
            return 'CPU'

    def _get_loss_and_metrics(self):
        """Возвращает функцию потерь и метрики в зависимости от типа задачи"""
        if self.task_type == 'classification':
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        else:  # regression
            loss = 'mse'  # Mean Squared Error
            metrics = ['mae', 'mse']  # Mean Absolute Error, Mean Squared Error

        return loss, metrics

    def _get_monitor_metric(self):
        """Возвращает метрику для мониторинга в callbacks"""
        if self.task_type == 'classification':
            return 'val_accuracy', 'max'
        else:  # regression
            return 'val_loss', 'min'

    def _create_callbacks(self, patience=50, reduce_lr_patience=5):
        """Создание callback'ов для обучения"""
        monitor_metric, monitor_mode = self._get_monitor_metric()

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=self.best_model_path,
                monitor=monitor_metric,
                mode=monitor_mode,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                save_freq='epoch'
            ),

            # keras.callbacks.EarlyStopping(
            #     monitor=monitor_metric,
            #     patience=patience,
            #     restore_best_weights=True,
            #     verbose=1,
            #     mode=monitor_mode
            # ),

            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1,
                mode=monitor_mode
            ),

            keras.callbacks.CSVLogger(
                os.path.join(self.checkpoint_dir, 'training_log.csv'),
                separator=',',
                append=True
            ),

            keras.callbacks.TerminateOnNaN()
        ]

        if self.device_type == 'GPU':
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.checkpoint_dir, 'tensorboard_logs'),
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch'
                )
            )

        return callbacks

    def _get_optimizer(self, initial_lr=0.01, optimizer_type='adam'):
        """Создание оптимизатора"""
        clipnorm = 1.0
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

    def _compile_model(self, optimizer):
        """Компиляция модели"""
        loss, metrics = self._get_loss_and_metrics()

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def set_y_scaler(self, scaler):
        """Установка scaler для нормализации целевых значений (для регрессии)"""
        self.scaler_y = scaler

    def train(self, X_train, y_train, X_val, y_val,
              batch_size=32, epochs=300, initial_lr=0.01,
              optimizer_type='adam', custom_lr_schedule=None,
              class_weight=None, validation_freq=1):
        """
        Обучение модели
        """
        print("\n" + "=" * 60)
        print(f"НАЧАЛО ОБУЧЕНИЯ НА {self.device_type} ({self.task_type.upper()})")
        print("=" * 60)

        print(f"Устройство: {self.device_type}")
        print(f"Размер батча: {batch_size}")
        print(f"Тренировочные данные: X {X_train.shape}, y {y_train.shape}")
        print(f"Валидационные данные: X {X_val.shape}, y {y_val.shape}")
        print(f"Эпохи: {epochs}")

        optimizer = self._get_optimizer(initial_lr, optimizer_type)
        self._compile_model(optimizer)

        if custom_lr_schedule is None:
            if self.task_type == 'classification':
                def lr_schedule(epoch, lr):
                    if epoch in [50, 150, 250]:
                        return lr * 0.1
                    return lr
            else:  # regression
                def lr_schedule(epoch, lr):
                    if epoch in [30, 50, 70]:
                        return lr * 0.1
                    return lr
        else:
            lr_schedule = custom_lr_schedule

        lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

        callbacks = self._create_callbacks()
        callbacks.append(lr_scheduler)

        start_time = time.time()

        try:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
                shuffle=True,
                class_weight=class_weight if self.task_type == 'classification' else None,
                validation_freq=validation_freq
            )

            self.training_time = time.time() - start_time

            print(f"\n✅ Обучение завершено за {self.training_time:.2f} секунд")
            print(f"✅ Лучшая модель сохранена в: {self.best_model_path}")

            self._save_training_info(X_train, X_val)

            return self.history

        except Exception as e:
            print(f"\n❌ Ошибка при обучении: {e}")
            print("Попытка восстановления...")

            if os.path.exists(self.best_model_path):
                self.model = keras.models.load_model(self.best_model_path)
                print("✅ Загружена лучшая сохраненная модель")

            raise

    def _save_training_info(self, X_train, X_val):
        """Сохранение информации об обучении"""
        info = {
            'task_type': self.task_type,
            'device_type': self.device_type,
            'training_time': self.training_time,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'input_shape': X_train.shape[1:],
            'model_summary': []
        }

        if self.task_type == 'classification':
            info.update({
                'num_classes': len(self.class_names),
                'class_names': self.class_names
            })
        else:  # regression
            info.update({
                'num_outputs': self.num_outputs,
                'output_names': self.output_names
            })

        self.model.summary(print_fn=lambda x: info['model_summary'].append(x))

        with open(self.metrics_file, 'w') as f:
            json.dump(info, f, indent=2, default=str)

        print(f"✅ Информация об обучении сохранена в {self.metrics_file}")

    def evaluate(self, X_test, y_test, batch_size=None):
        """
        Оценка модели на тестовых данных
        """
        print("\n" + "=" * 60)
        print(f"ОЦЕНКА МОДЕЛИ ({self.task_type.upper()})")
        print("=" * 60)

        if batch_size is None:
            batch_size = 32

        # Базовая оценка
        results = self.model.evaluate(
            X_test, y_test,
            batch_size=batch_size,
            verbose=0,
            return_dict=True
        )

        print(f"\n📊 Базовые метрики:")
        for metric_name, metric_value in results.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        # Предсказания
        y_pred = self.model.predict(X_test, batch_size=batch_size, verbose=0)

        # Специфичная оценка в зависимости от типа задачи
        if self.task_type == 'classification':
            self._evaluate_classification(y_test, y_pred)
        else:  # regression
            self._evaluate_regression(y_test, y_pred, self.scaler_y)

        # Сохранение метрик
        self.metrics = {
            'task_type': self.task_type,
            'test_metrics': results,
            'device_used': self.device_type,
            'evaluation_time': time.time()
        }

        if self.task_type == 'classification':
            self.metrics.update(self._get_classification_metrics_dict())
        else:
            self.metrics.update(self._get_regression_metrics_dict(y_test, y_pred))

        with open(f'evaluation_metrics_{self.task_type}.json', 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        return self.metrics

    def _evaluate_classification(self, y_test, y_pred):
        """Оценка для классификации"""
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        try:
            if len(self.class_names) == 2:
                roc_auc = roc_auc_score(y_true, y_pred[:, 1])
                print(f"\n📈 ROC AUC Score: {roc_auc:.4f}")
            else:
                roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
                print(f"\n📈 ROC AUC Score (macro): {roc_auc:.4f}")
        except:
            roc_auc = None
            print("\n⚠️ ROC AUC не может быть вычислен")

        print("\n📊 Classification Report:")
        print(classification_report(
            y_true, y_pred_classes,
            target_names=self.class_names,
            digits=4
        ))

    def _evaluate_regression(self, y_test, y_pred, scaler=None):
        # Всегда приводим к оригинальному масштабу для метрик
        if scaler is not None:
            y_test_real = scaler.inverse_transform(y_test)
            y_pred_real = scaler.inverse_transform(y_pred)
        else:
            y_test_real = y_test
            y_pred_real = y_pred

        print("\n📊 Регрессионные метрики (в исходном масштабе):")

        for i in range(self.num_outputs):
            output_name = self.output_names[i] if i < len(self.output_names) else f'Output {i}'

            # Считаем ВСЁ на реальных данных
            mse = mean_squared_error(y_test_real[:, i], y_pred_real[:, i])
            mae = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
            r2 = r2_score(y_test_real[:, i], y_pred_real[:, i])

            print(f"\n  {output_name}:")
            print(f"    MSE:  {mse:.4e}") # Используем экспоненту для больших чисел
            print(f"    MAE:  {mae:.4e}")
            print(f"    R²:   {r2:.4f}")

            # Диагностика: если R2 < 0, модель хуже, чем простое среднее
            if r2 < 0:
                print(f"    ⚠️  КРИТИЧЕСКАЯ ОШИБКА: Модель работает хуже константы (среднего)!")

    def _get_regression_metrics_dict(self, y_test, y_pred):
        """Возвращает словарь с метриками регрессии"""
        if self.scaler_y is not None:
            y_test_orig = self.scaler_y.inverse_transform(y_test)
            y_pred_orig = self.scaler_y.inverse_transform(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred

        metrics_dict = {
            'output_names': self.output_names,
            'per_output_metrics': {}
        }

        for i in range(self.num_outputs):
            mse = float(mean_squared_error(y_test_orig[:, i], y_pred_orig[:, i]))
            mae = float(mean_absolute_error(y_test_orig[:, i], y_pred_orig[:, i]))
            r2 = float(r2_score(y_test_orig[:, i], y_pred_orig[:, i]))

            metrics_dict['per_output_metrics'][self.output_names[i]] = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }

        metrics_dict['overall'] = {
            'mse': float(mean_squared_error(y_test_orig, y_pred_orig)),
            'mae': float(mean_absolute_error(y_test_orig, y_pred_orig))
        }

        return metrics_dict

    def _get_classification_metrics_dict(self):
        """Возвращает словарь с метриками классификации"""
        return {}  # Здесь можно добавить сохранение classification report в словарь

    def predict_single(self, X_single, return_original_scale=False):
        """
        Предсказание для одного примера

        Parameters:
        -----------
        X_single : numpy array
            Один пример данных
        return_original_scale : bool
            Для регрессии - возвращать ли значения в исходном масштабе

        Returns:
        --------
        Для классификации: (class_name, probabilities, confidence)
            - class_name : str - название предсказанного класса
            - probabilities : array - вероятности для всех классов
            - confidence : float - уверенность в предсказании
        Для регрессии: predictions (и опционально в исходном масштабе)
        """
        X_batch = np.expand_dims(X_single, axis=0)
        predictions = self.model.predict(X_batch, batch_size=1, verbose=0)[0]

        if self.task_type == 'classification':
            predicted_class_idx = np.argmax(predictions)
            confidence = predictions[predicted_class_idx]

            # Получаем имя класса
            if self.class_names and predicted_class_idx < len(self.class_names):
                class_name = self.class_names[predicted_class_idx]
            else:
                class_name = f"Class_{predicted_class_idx}"

            return class_name, predictions, confidence
        else:  # regression
            if return_original_scale and self.scaler_y is not None:
                predictions = predictions.reshape(1, -1)
                predictions = self.scaler_y.inverse_transform(predictions)[0]
            return predictions

    def plot_training_history(self, save_path=None):
        """Визуализация истории обучения"""
        if self.history is None:
            print("⚠️ Модель еще не обучена!")
            return

        history_dict = self.history.history

        if self.task_type == 'classification':
            self._plot_classification_history(history_dict, save_path)
        else:
            self._plot_regression_history(history_dict, save_path)

    def _plot_classification_history(self, history_dict, save_path):
        """Визуализация истории для классификации"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        axes[0].plot(history_dict['accuracy'], label='Train', linewidth=2)
        axes[0].plot(history_dict['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title(f'Model Accuracy ({self.device_type})', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
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

    def _plot_regression_history(self, history_dict, save_path):
        """Визуализация истории для регрессии"""
        num_metrics = len([k for k in history_dict.keys() if not k.startswith('val_')])
        val_metrics = len([k for k in history_dict.keys() if k.startswith('val_')])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(history_dict['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title(f'Model Loss ({self.device_type})', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE если есть
        if 'mae' in history_dict:
            axes[1].plot(history_dict['mae'], label='Train MAE', linewidth=2)
            axes[1].plot(history_dict['val_mae'], label='Validation MAE', linewidth=2)
            axes[1].set_title(f'Model MAE ({self.device_type})', fontsize=14)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 График сохранен в {save_path}")

        plt.show()

    def plot_confusion_matrix(self, X_test, y_test, batch_size=None,
                             normalize=True, save_path=None, show_plot=True):
        """
        Построение и визуализация confusion matrix

        Parameters:
        -----------
        X_test, y_test : numpy arrays
            Тестовые данные
        batch_size : int, optional
            Размер батча для предсказания
        normalize : bool
            Нормализовать ли значения (проценты) или оставить абсолютные значения
        save_path : str, optional
            Путь для сохранения графика
        show_plot : bool
            Показывать ли график

        Returns:
        --------
        numpy.ndarray : confusion matrix
        """
        print("\n" + "=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)

        if batch_size is None:
            batch_size = 32

        y_pred = self.model.predict(X_test, batch_size=batch_size, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred_classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            fmt = '.2f'
            title_suffix = ' (%)'
        else:
            fmt = 'd'
            title_suffix = ''

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               title=f'Confusion Matrix{title_suffix}',
               ylabel='True Label',
               xlabel='Predicted Label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10)

        fig.tight_layout()

        if self.metrics is not None:
            self.metrics['confusion_matrix'] = cm.tolist()
            self.metrics['confusion_matrix_normalized'] = normalize
            with open('evaluation_metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Confusion matrix сохранена в {save_path}")

        if show_plot:
            plt.show()

        return cm


    def plot_predictions_vs_actual(self, X_test, y_test, batch_size=None, save_path=None):
        """
        Построение графика предсказаний vs реальных значений (только для регрессии)
        """
        if self.task_type != 'regression':
            print("⚠️ Этот метод доступен только для регрессии")
            return

        if batch_size is None:
            batch_size = 32

        y_pred = self.model.predict(X_test, batch_size=batch_size, verbose=0)

        if self.scaler_y is not None:
            y_test_orig = self.scaler_y.inverse_transform(y_test)
            y_pred_orig = self.scaler_y.inverse_transform(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred

        print(f"Количество выходов: {self.num_outputs}")
        print(f"Форма y_test_orig: {y_test_orig.shape}")
        print("Results:", y_pred_orig[:, 0].min(), y_pred_orig[:, 0].max())
        print("Results:", y_test_orig[:, 0].min(), y_test_orig[:, 0].max())

        # Отображаем ВСЕ выходы, но не более 6 для читаемости
        n_outputs = self.num_outputs  # Убираем min, чтобы видеть все выходы
        max_display = 8  # Максимум для отображения

        if n_outputs > max_display:
            print(f"⚠️ Внимание: {n_outputs} выходов, но для читаемости будет показано только {max_display}")
            n_outputs = max_display

        n_cols = min(n_outputs, 2)
        n_rows = (n_outputs + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        if n_outputs == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i in range(n_outputs):
            ax = axes[i]
            output_name = self.output_names[i] if i < len(self.output_names) else f'Output {i}'

            ax.scatter(y_test_orig[:, i], y_pred_orig[:, i], alpha=0.5, s=10)

            # Линия идеального предсказания
            min_val = min(y_test_orig[:, i].min(), y_pred_orig[:, i].min())
            max_val = max(y_test_orig[:, i].max(), y_pred_orig[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное предсказание')

            ax.set_xlabel('Реальные значения')
            ax.set_ylabel('Предсказанные значения')
            ax.set_title(f'{output_name}\nR² = {r2_score(y_test_orig[:, i], y_pred_orig[:, i]):.4f}')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Скрыть лишние подграфики
        for j in range(n_outputs, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 График сохранен в {save_path}")

        plt.show()

    def get_bad_predictions(self, X_test, y_test, batch_size=None, error_threshold=None, top_k=None):
        """
        Находит примеры с плохими предсказаниями

        Параметры:
        -----------
        error_threshold : float, optional
            Порог абсолютной ошибки
        top_k : int, optional
            Количество самых плохих предсказаний
        """
        if self.task_type != 'regression':
            print("⚠️ Этот метод доступен только для регрессии")
            return None

        if batch_size is None:
            batch_size = 32

        y_pred = self.model.predict(X_test, batch_size=batch_size, verbose=0)

        if self.scaler_y is not None:
            y_test_orig = self.scaler_y.inverse_transform(y_test)
            y_pred_orig = self.scaler_y.inverse_transform(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred

        # Отладочная информация
        print(f"Shape of y_test_orig: {y_test_orig.shape}")
        print(f"Shape of y_pred_orig: {y_pred_orig.shape}")
        print(f"Type of X_test: {type(X_test)}")
        if hasattr(X_test, 'shape'):
            print(f"Shape of X_test: {X_test.shape}")

        # Убеждаемся, что данные - numpy массивы
        y_test_orig = np.array(y_test_orig)
        y_pred_orig = np.array(y_pred_orig)

        # Проверяем размерности y
        if y_test_orig.ndim == 1:
            y_test_orig = y_test_orig.reshape(-1, 1)
            y_pred_orig = y_pred_orig.reshape(-1, 1)

        # Вычисляем ошибки
        abs_errors = np.abs(y_test_orig - y_pred_orig)
        rel_errors = np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 1e-8)) * 100

        # Создаем DataFrame с результатами
        results_data = {}

        # Добавляем информацию о предсказаниях и ошибках для каждого выхода
        n_outputs = y_test_orig.shape[1]
        for i in range(n_outputs):
            output_name = self.output_names[i] if hasattr(self, 'output_names') and i < len(self.output_names) else f'Output_{i}'
            results_data[f'Actual_{output_name}'] = y_test_orig[:, i]
            results_data[f'Predicted_{output_name}'] = y_pred_orig[:, i]
            results_data[f'AbsError_{output_name}'] = abs_errors[:, i]
            results_data[f'RelError_{output_name}_%'] = rel_errors[:, i]

        # Добавляем общие метрики
        results_data['MeanAbsError'] = abs_errors.mean(axis=1)
        results_data['MeanRelError_%'] = rel_errors.mean(axis=1)
        results_data['MaxAbsError'] = abs_errors.max(axis=1)

        # Создаем DataFrame из результатов
        results_df = pd.DataFrame(results_data)

        # Для 3-мерных данных X_test (например, для LSTM) добавляем сводные статистики
        if X_test.ndim == 3:
            print("Обнаружены 3-мерные данные, добавляем статистики признаков...")

            # Вычисляем статистики для каждого временного шага
            # Форма: (samples, time_steps, features)
            X_test_array = np.array(X_test)

            # Добавляем статистики по временному измерению
            results_data_features = {}

            for i in range(X_test_array.shape[2]):  # По количеству признаков
                feature_data = X_test_array[:, :, i]
                results_data_features[f'Feature_{i}_mean'] = feature_data.mean(axis=1)
                results_data_features[f'Feature_{i}_std'] = feature_data.std(axis=1)
                results_data_features[f'Feature_{i}_min'] = feature_data.min(axis=1)
                results_data_features[f'Feature_{i}_max'] = feature_data.max(axis=1)
                results_data_features[f'Feature_{i}_last'] = feature_data[:, -1]  # Последнее значение

            # Создаем DataFrame с признаками
            features_df = pd.DataFrame(results_data_features)

            # Объединяем с результатами
            results_df = pd.concat([features_df, results_df], axis=1)

        elif X_test.ndim == 2:
            # Для 2-мерных данных добавляем все признаки
            X_test_array = np.array(X_test)
            for i in range(X_test_array.shape[1]):
                results_df[f'Feature_{i}'] = X_test_array[:, i]

        elif isinstance(X_test, pd.DataFrame):
            # Для DataFrame
            for col in X_test.columns:
                results_df[col] = X_test[col].values

        print(f"Created DataFrame with shape: {results_df.shape}")

        # Фильтруем по порогу
        if error_threshold is not None:
            mask = results_df['MeanAbsError'] > error_threshold
            bad_predictions = results_df[mask].copy()
            print(f"Found {len(bad_predictions)} bad predictions with error > {error_threshold}")
        else:
            bad_predictions = results_df.copy()

        # Берем топ-K
        if top_k is not None:
            bad_predictions = bad_predictions.nlargest(top_k, 'MeanAbsError')
            print(f"Taking top {top_k} bad predictions")

        # Сортируем по убыванию ошибки
        bad_predictions = bad_predictions.sort_values('MeanAbsError', ascending=False)

        # Добавляем информацию о процентиле
        if len(bad_predictions) > 0:
            bad_predictions['ErrorPercentile'] = bad_predictions['MeanAbsError'].rank(pct=True) * 100

        return bad_predictions

    def save_model(self, path=None, format='keras'):
        """Сохранение модели"""
        if path is None:
            path = f'welltest_model_{self.task_type}'

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        if format in ['h5', 'all']:
            try:
                weights_path = f"{path}.weights.h5"
                self.model.save_weights(weights_path)
                print(f"✅ Веса модели сохранены: {weights_path}")
            except Exception as e:
                print(f"❌ Не удалось сохранить веса: {e}")

        if format in ['keras', 'all']:
            keras_path = f"{path}.keras" if not path.endswith('.keras') else path
            try:
                self.model.save(keras_path)
                print(f"✅ Модель сохранена в Keras format: {keras_path}")
            except Exception as e:
                print(f"❌ Не удалось сохранить в Keras format: {e}")

        if format in ['savedmodel', 'all']:
            savedmodel_path = f"{path}_savedmodel"
            try:
                tf.saved_model.save(self.model, savedmodel_path)
                print(f"✅ Модель сохранена в SavedModel: {savedmodel_path}")
            except Exception as e:
                print(f"❌ Не удалось сохранить в SavedModel: {e}")

        # Сохранение конфигурации
        config = {
            'task_type': self.task_type,
            'class_names': self.class_names if self.task_type == 'classification' else [],
            'output_names': self.output_names if self.task_type == 'regression' else [],
            'num_outputs': self.num_outputs if self.task_type == 'regression' else None,
            'input_shape': self.model.input_shape[1:],
            'device_used': self.device_type,
            'model_format': format,
            'save_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        config_path = f"{path}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Конфигурация сохранена: {config_path}")