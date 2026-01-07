from tensorflow import keras
from tensorflow.keras import layers


class WellTest1DCNN:
    def __init__(self, num_classes=2, input_shape=(128, 2)):
        """
        Инициализация 1D CNN для классификации кривых ГДИ

        Parameters:
        -----------
        num_classes : int
            Количество классов (типов пластов)
        input_shape : tuple
            Форма входных данных (каналы, временные точки)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """Построение архитектуры сети как в статье"""
        inputs = keras.Input(shape=self.input_shape)  # (2, 128)

        # Первый сверточный блок
        x = layers.Conv1D(
            filters=32,
            kernel_size=5,
            strides=2,
            padding='same',
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

        # Второй сверточный блок
        x = layers.Conv1D(
            filters=64,
            kernel_size=5,
            strides=2,
            padding='same',
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

        # Третий сверточный блок
        x = layers.Conv1D(
            filters=128,
            kernel_size=5,
            strides=2,
            padding='same',
            name='conv1d_3'
        )(x)
        x = layers.BatchNormalization(name='bn1d_3')(x)
        x = layers.ReLU(name='relu_3')(x)
        x = layers.MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same',
            name='maxpool1d_3'
        )(x)

        # Полносвязные слои
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(256, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        x = layers.Dense(128, activation='relu', name='fc2')(x)
        x = layers.Dropout(0.5, name='dropout2')(x)

        # Выходной слой
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)

        # Создание модели
        model = keras.Model(inputs=inputs, outputs=outputs, name='WellTest1DCNN')

        return model

    def compile_model(self, learning_rate=0.001):
        """Компиляция модели с оптимайзером Adam"""
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def summary(self):
        """Вывод архитектуры модели"""
        return self.model.summary()