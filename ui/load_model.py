import numpy as np
import pandas as pd

from tensorflow import keras

def normalize(X):
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


def load_cls_model(cls_model: str, ):
	print('Загрузка модели')

	if cls_model == '1d_cnn':
		model = keras.models.load_model('/home/ilnaz/PycharmProjects/data-gen/ui/models/1d_cnn_model_classification_multiclass.keras')

	print("Модель классификации успешно загружена!")

	# homogeneous_reg_inf_model = keras.models.load_model('/home/ilnaz/PycharmProjects/data-gen/ui/models/1d_cnn_model_regression_homogeneous_inf.keras')
	
	return model

def predict(model, df, k):
	t_D = df['t_D']
	y = df['P_wD_gauss']
	dy = df['dP_wD']

	t_D = t_D / k
	dy = dy / k

	data = pd.DataFrame({'t_D': t_D, 'dy': dy})

	data_norm = normalize(data)

	# 1. Приводим данные к нужному формату (batch_size, timesteps, channels)
	# Предполагаем, что data_norm имеет форму (timesteps, 2)
	input_data = np.expand_dims(data_norm.values, axis=0).astype('float32')

	# 2. Делаем предсказание напрямую через keras модель
	predictions = model.predict(input_data)

	# 3. Обрабатываем результат (argmax для классификации)
	predicted_class_idx = np.argmax(predictions, axis=1)[0]

	# Список имен классов (должен совпадать с тем, на чем обучали)
	class_names = [
	    'dual_permeability_inf', 'homogeneous_fin', 'dual_porosity_inf',
	    'dual_porosity_fin', 'radial_composite_inf', 'homogeneous_inf',
	    'dual_permeability_fin'
	]

	return class_names[predicted_class_idx]