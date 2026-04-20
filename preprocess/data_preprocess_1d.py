import os
import re

import numpy as np
import pandas as pd
from typing_extensions import override

from preprocess.multilabel_cls import MultilabelEncoder
from preprocess.abstract_preprcessor import AbstractPressureDataClassificationPreprocessor, \
    AbstractPressureDataPreprocessor, Model


class PressureDataClassificationPreprocessor1D(AbstractPressureDataClassificationPreprocessor):
    """Подготовка данных для классификации типа пласта из файлов кривых"""

    def __init__(self, data_dir, test_size=0.2, val_size=0.1, random_state=42, debug=False):
        super().__init__(data_dir, test_size, val_size, random_state, debug)

    @override
    def load_data(self):
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


class PressureDataRegressionPreprocessor1D(AbstractPressureDataPreprocessor):

    def __init__(self, data_dir, model_type, test_size=0.2, val_size=0.1, random_state=42, debug=False):
        super().__init__(data_dir, test_size, val_size, random_state, debug)

        self.model_type = model_type

    @override
    def load_data(self):
        """
        Загрузка датасета
        :return: np.array(X), np.array(y)
        """
        X_list = []
        y_list = []

        curve_data_dir = self.data_dir + "/curve"

        pattern = r'(.+)_(\d+)\.csv$'

        for item in os.listdir(curve_data_dir):
            match = re.match(pattern, item)
            model_type = Model(match.group(1))
            num = int(match.group(2))

            if self.model_type != model_type:
                continue

            file_path = os.path.join(curve_data_dir, item)

            df = pd.read_csv(file_path)[['t_D', 'dP_wD']]

            t_D = np.array(df['t_D'])
            dP_wD = np.array(df['dP_wD'])

            sample = np.column_stack([dP_wD, t_D])
            X_list.append(sample)

            y_list.append(self.get_params_from_file(num))

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y

    def get_params_from_file(self, num):
        file_name = f'{self.data_dir}/params/{num}.csv'
        param_df = pd.read_csv(file_name)

        if self.model_type == Model.HOMOGENEOUS_INF:
            return [param_df['k'], param_df['C_D'], param_df['S']]

        if self.model_type == Model.HOMOGENEOUS_FIN:
            return [param_df['k'], param_df['C_D'], param_df['S'], param_df['r_D_e']]

        if self.model_type == Model.DUAL_POROSITY_INF:
            return [param_df['k_f'], param_df['C_D'], param_df['S'], param_df['omega'], param_df['lambda']]

        if self.model_type == Model.DUAL_POROSITY_FIN:
            return [param_df['k_f'], param_df['C_D'], param_df['S'], param_df['omega'], param_df['lambda'], param_df['R_eD']]

        if self.model_type == Model.DUAL_PERMEABILITY_INF:
            return [param_df['k2'], param_df['C_D'], param_df['S2'], param_df['omega'], param_df['lambda'], param_df['kappa']]

        if self.model_type == Model.DUAL_PERMEABILITY_FIN:
            return [param_df['k2'], param_df['C_D'], param_df['S2'], param_df['omega'], param_df['lambda'], param_df['kappa'], param_df['R_eD']]

        if self.model_type == Model.RADIAL_COMPOSITE_INF:
            return [param_df['k1'],
                    param_df['C_D'],
                    param_df['S'],
                    param_df['M1'],
                    param_df['M2'],
                    param_df['omega1'],
                    param_df['omega2'],
                    param_df['r_fD']]

        return None


class PressureDataMultilabelClassificationPreprocessor1D(AbstractPressureDataClassificationPreprocessor):
    """Подготовка данных для multilabel классификации типа пласта из файлов кривых"""

    def __init__(self, data_dir, test_size=0.2, val_size=0.1, random_state=42, debug=False):
        super().__init__(data_dir, test_size, val_size, random_state, debug)
        self.encoder = MultilabelEncoder()

    @override
    def load_data(self):
        """
        Загрузка датасета
        :return: np.array(X), np.array(y)
        """
        X_list = []
        y_list = []

        curve_data_dir = self.data_dir + "/curve"

        for item in os.listdir(curve_data_dir):
            file_path = os.path.join(curve_data_dir, item)

            df = pd.read_csv(file_path)[['t_D', 'dP_wD']]

            t_D = np.array(df['t_D'])
            dP_wD = np.array(df['dP_wD'])

            model, boundary, num, type = self._get_labels_from_filename(item)
            num = int(num)

            k = self.get_k(model, num)[0]

            dP_wD = dP_wD / k
            t_D = t_D / k

            sample = np.column_stack([dP_wD, t_D])
            X_list.append(sample)

            target = self.encoder.encode(model, boundary, type)
            y_list.append(target)

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y

    def _get_labels_from_filename(self, item):
        """
        Получение target из названия файла.
        Пример входного файла: homogeneous_inf_12_inc1
        """
        basename = os.path.basename(item)
        name_without_ext = os.path.splitext(basename)[0]
        parts = name_without_ext.split('_')

        if len(parts) > 4:
            return parts[0] + '_' + parts[1], parts[2], parts[3], parts[4]
        else:
            return parts[0], parts[1], parts[2], parts[3]

    def get_k(self, model_type, num):
        """
        Получение параметра k.
        """
        file_name = f'{self.data_dir}/params/{num}.csv'
        param_df = pd.read_csv(file_name)

        if model_type == 'homogeneous':
            return param_df['k']

        if model_type == 'dual_porosity':
            return param_df['k_f']

        if model_type == 'dual_permeability':
            return param_df['k2']

        if model_type == 'radial_composite':
            return param_df['k1']

        return None


if __name__ == "__main__":
    # data_preprocess = PressureDataClassificationPreprocessor1D(data_dir='../datasets/dataset/curve', debug=True)
    # X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()
    #
    # data_preprocess.stats()

    # data_preprocess = PressureDataRegressionPreprocessor1D(data_dir='../datasets/dataset',
    #                                                        model_type=Model.HOMOGENEOUS_FIN)

    # data_preprocess = PressureDataRegressionPreprocessor1D(data_dir='../datasets/dataset',
    #                                                        model_type=Model.DUAL_POROSITY_INF)

    # data_preprocess = PressureDataRegressionPreprocessor1D(data_dir='../datasets/dataset',
    #                                                        model_type=Model.DUAL_POROSITY_FIN)

    # data_preprocess = PressureDataRegressionPreprocessor1D(data_dir='../datasets/dataset',
    #                                                        model_type=Model.DUAL_PERMEABILITY_INF)

    # data_preprocess = PressureDataRegressionPreprocessor1D(data_dir='../datasets/dataset',
    #                                                        model_type=Model.DUAL_PERMEABILITY_FIN)

    # data_preprocess = PressureDataRegressionPreprocessor1D(data_dir='../datasets/dataset',
    #                                                        model_type=Model.HOMOGENEOUS_INF,
    #                                                        debug=True)

    data_preprocess = PressureDataMultilabelClassificationPreprocessor1D(data_dir='/home/ilnaz/PycharmProjects/datasets/models_incs_prc')

    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset(task='classification')

    print('y_shape', y_train[0].shape)
    encoder = MultilabelEncoder()
    print(encoder.decode(y_train[19000]))