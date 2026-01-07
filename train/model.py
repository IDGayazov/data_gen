# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np


# class WellTest1DCNN:
#     def __init__(self, num_classes=3, input_shape=(2, 128)):
#         """
#         Инициализация 1D CNN для классификации кривых ГДИ

#         Parameters:
#         -----------
#         num_classes : int
#             Количество классов (типов пластов)
#         input_shape : tuple
#             Форма входных данных (каналы, временные точки)
#         """
#         self.num_classes = num_classes
#         self.input_shape = input_shape
#         self.model = self._build_model()

#     def _build_model(self):
#         """Построение архитектуры сети как в статье"""
#         inputs = keras.Input(shape=self.input_shape)  # (2, 128)

#         # Первый сверточный блок
#         x = layers.Conv1D(
#             filters=32,
#             kernel_size=5,
#             strides=2,
#             padding='same',
#             name='conv1d_1'
#         )(inputs)
#         x = layers.BatchNormalization(name='bn1d_1')(x)
#         x = layers.ReLU(name='relu_1')(x)
#         x = layers.MaxPooling1D(
#             pool_size=2,
#             strides=2,
#             padding='same',
#             name='maxpool1d_1'
#         )(x)

#         # Второй сверточный блок
#         x = layers.Conv1D(
#             filters=64,
#             kernel_size=5,
#             strides=2,
#             padding='same',
#             name='conv1d_2'
#         )(x)
#         x = layers.BatchNormalization(name='bn1d_2')(x)
#         x = layers.ReLU(name='relu_2')(x)
#         x = layers.MaxPooling1D(
#             pool_size=2,
#             strides=2,
#             padding='same',
#             name='maxpool1d_2'
#         )(x)

#         # Третий сверточный блок
#         x = layers.Conv1D(
#             filters=128,
#             kernel_size=5,
#             strides=2,
#             padding='same',
#             name='conv1d_3'
#         )(x)
#         x = layers.BatchNormalization(name='bn1d_3')(x)
#         x = layers.ReLU(name='relu_3')(x)
#         x = layers.MaxPooling1D(
#             pool_size=2,
#             strides=2,
#             padding='same',
#             name='maxpool1d_3'
#         )(x)

#         # Полносвязные слои
#         x = layers.Flatten(name='flatten')(x)
#         x = layers.Dense(256, activation='relu', name='fc1')(x)
#         x = layers.Dropout(0.5, name='dropout1')(x)
#         x = layers.Dense(128, activation='relu', name='fc2')(x)
#         x = layers.Dropout(0.5, name='dropout2')(x)

#         # Выходной слой
#         outputs = layers.Dense(
#             self.num_classes,
#             activation='softmax',
#             name='output'
#         )(x)

#         # Создание модели
#         model = keras.Model(inputs=inputs, outputs=outputs, name='WellTest1DCNN')

#         return model

#     def compile_model(self, learning_rate=0.001):
#         """Компиляция модели с оптимайзером Adam"""
#         optimizer = keras.optimizers.Adam(
#             learning_rate=learning_rate,
#             beta_1=0.9,
#             beta_2=0.999,
#             epsilon=1e-8
#         )

#         self.model.compile(
#             optimizer=optimizer,
#             loss='categorical_crossentropy',
#             metrics=['accuracy']
#         )

#         return self.model

#     def summary(self):
#         """Вывод архитектуры модели"""
#         return self.model.summary()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class WellTest1DCNN:
    def __init__(self, num_classes=3, input_shape=(128, 2)):
        """
        Инициализация 1D CNN для классификации кривых ГДИ
        
        Parameters:
        -----------
        num_classes : int
            Количество классов (типов пластов)
        input_shape : tuple
            Форма входных данных (временные точки, каналы)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """Построение улучшенной архитектуры сети"""
        inputs = keras.Input(shape=self.input_shape)  # (128, 2)
        
        # ========== БЛОК 1: Извлечение низкоуровневых признаков ==========
        x = layers.Conv1D(
            filters=32,
            kernel_size=5,
            strides=1,  # Убрали strides=2
            padding='same',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='conv1d_1'
        )(inputs)
        x = layers.BatchNormalization(name='bn1d_1')(x)
        x = layers.ReLU(name='relu_1')(x)
        x = layers.MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same',
            name='maxpool1d_1'
        )(x)
        x = layers.Dropout(0.2, name='dropout_conv1')(x)  # Добавили Dropout
        
        # ========== БЛОК 2: Извлечение среднего уровня признаков ==========
        x = layers.Conv1D(
            filters=64,
            kernel_size=5,
            strides=1,  # Убрали strides=2
            padding='same',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='conv1d_2'
        )(x)
        x = layers.BatchNormalization(name='bn1d_2')(x)
        x = layers.ReLU(name='relu_2')(x)
        x = layers.MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same',
            name='maxpool1d_2'
        )(x)
        x = layers.Dropout(0.3, name='dropout_conv2')(x)
        
        # ========== БЛОК 3: Извлечение высокоуровневых признаков ==========
        x = layers.Conv1D(
            filters=128,
            kernel_size=3,  # Уменьшили kernel_size
            strides=1,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='conv1d_3'
        )(x)
        x = layers.BatchNormalization(name='bn1d_3')(x)
        x = layers.ReLU(name='relu_3')(x)
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)  # Заменили MaxPool
        
        # ========== ПОЛНОСВЯЗНЫЙ БЛОК ==========
        x = layers.Dense(256, activation='relu', name='fc1')(x)
        x = layers.BatchNormalization(name='bn_fc1')(x)
        x = layers.Dropout(0.5, name='dropout_fc1')(x)
        
        x = layers.Dense(128, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization(name='bn_fc2')(x)
        x = layers.Dropout(0.5, name='dropout_fc2')(x)
        
        # Выходной слой
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        # Создание модели
        model = keras.Model(inputs=inputs, outputs=outputs, name='WellTest1DCNN')
        
        return model
    
    def compile_model(self, learning_rate=0.001, class_weights=None):
        """Компиляция модели с учетом дисбаланса классов"""
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.0001,  # L2 регуляризация
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Используем Focal Loss для дисбалансированных данных
        if class_weights is not None:
            loss = self._weighted_categorical_crossentropy(class_weights)
        else:
            loss = 'categorical_crossentropy'
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        return self.model
    
    def _weighted_categorical_crossentropy(self, class_weights):
        """Взвешенная категориальная кросс-энтропия"""
        class_weights_tensor = tf.constant(list(class_weights.values()), dtype=tf.float32)
        
        def loss(y_true, y_pred):
            # Стандартная кросс-энтропия
            ce = keras.losses.categorical_crossentropy(y_true, y_pred)
            
            # Взвешивание
            y_true_class = tf.argmax(y_true, axis=1)
            weights = tf.gather(class_weights_tensor, y_true_class)
            
            return tf.reduce_mean(ce * weights)
        
        return loss

    def summary(self):
        """Вывод архитектуры модели"""
        return self.model.summary()