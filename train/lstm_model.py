from tensorflow import keras
from tensorflow.keras import layers


class WellTestLSTM:
    def __init__(self, num_classes=2, input_shape=(128, 2)):
        """
        Инициализация LSTM сети для классификации кривых ГДИС

        Parameters:
        -----------
        num_classes : int
            Количество классов (типов пластов)
        input_shape : tuple
            Форма входных данных (временные точки, признаки)
            Для LSTM обычно (timesteps, features)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape  # (128, 2) - 128 временных шагов, 2 признака
        self.model = self._build_model()

    def _build_model(self):
        """Построение архитектуры сети с LSTM"""
        inputs = keras.Input(shape=self.input_shape)  # (128, 2)

        # Первый LSTM блок (возвращает последовательность для следующего LSTM)
        x = layers.LSTM(
            units=64,
            return_sequences=True,
            name='lstm_1'
        )(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(0.3, name='dropout_1')(x)

        # Второй LSTM блок
        x = layers.LSTM(
            units=128,
            return_sequences=True,
            name='lstm_2'
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)

        # Третий LSTM блок (возвращает только последний выход)
        x = layers.LSTM(
            units=256,
            return_sequences=False,
            name='lstm_3'
        )(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.Dropout(0.3, name='dropout_3')(x)

        # Полносвязные слои
        x = layers.Dense(128, activation='relu', name='fc1')(x)
        x = layers.BatchNormalization(name='bn_fc1')(x)
        x = layers.Dropout(0.5, name='dropout_fc1')(x)

        x = layers.Dense(64, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization(name='bn_fc2')(x)
        x = layers.Dropout(0.5, name='dropout_fc2')(x)

        # Выходной слой
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='WellTestLSTM')

        return model

    def summary(self):
        """Вывод архитектуры модели"""
        return self.model.summary()
