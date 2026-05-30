import os
import sys

import numpy as np
import onnxruntime as ort
import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QMessageBox
from PyQt5.QtCore import Qt

from ui.util import (
    predict,
    predict_params,
    build_predicted_curve,
    get_regression_model_paths,
    format_params,
)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

_CLS_MODELS = {
    'cnn': (
        '1d_cnn_model_classification_multiclass.onnx',
        '1d_cnn_model_classification_multiclass_config.json',
    ),
    'cnn_lstm': (
        'cnn_lstm_cls_mutliclass.onnx',
        'cnn_lstm_cls_mutliclass_config.json',
    ),
    'lstm': (
        'lstm_model_classification_multiclass.onnx',
        'lstm_model_classification_multiclass_config.json',
    ),
}


class GenerationWindow(QDialog):
    def __init__(self, dataset=None, parent=None):
        super().__init__(parent)
        ui_path = os.path.join(os.path.dirname(__file__), 'prediction.ui')
        uic.loadUi(ui_path, self)

        self.dataset = dataset
        self.model_name = ""
        self.k = 1

        if hasattr(self, 'prediction'):
            self.prediction.clicked.connect(self.predict)

        self.setup_loglog_plot()

    # ------------------------------------------------------------------ #
    # Plot setup
    # ------------------------------------------------------------------ #

    def setup_loglog_plot(self):
        if not hasattr(self, 'graphicsView'):
            QMessageBox.warning(self, "Ошибка", "В UI файле отсутствует graphicsView!")
            return

        layout = QVBoxLayout(self.graphicsView)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setLogMode(x=True, y=True)
        self.plot_widget.setLabel('left', 'Значение', units='')
        self.plot_widget.setLabel('bottom', 'Время', units='')
        self.plot_widget.setTitle('Логарифмический график (log-log)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setBackground('white')
        self.plot_widget.addLegend()

    # ------------------------------------------------------------------ #
    # Main prediction pipeline
    # ------------------------------------------------------------------ #

    def predict(self):
        # --- 1. Identify which classification model is selected ----------
        if self.cnn.isChecked():
            model_key = 'cnn'
        elif self.cnn_lstm.isChecked():
            model_key = 'cnn_lstm'
        else:
            model_key = 'lstm'

        onnx_file, cfg_file = _CLS_MODELS[model_key]
        cls_onnx = os.path.join(_MODELS_DIR, onnx_file)
        cls_cfg = os.path.join(_MODELS_DIR, cfg_file)

        # --- 2. Classification -------------------------------------------
        try:
            cls_session = ort.InferenceSession(cls_onnx)
            cls_name = predict(cls_session, self.dataset, self.k, cls_cfg)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка классификации", str(e))
            return

        self.lineEdit.setText(cls_name)

        # --- 3. Regression: load model + scaler config -------------------
        reg_onnx, reg_scaler = get_regression_model_paths(cls_name, _MODELS_DIR, model_key)

        if reg_onnx is None or not os.path.exists(reg_onnx):
            self._show_params("Регрессионная модель не найдена для типа: " + cls_name)
            return

        if not os.path.exists(reg_scaler):
            self._show_params(
                "Конфиг скейлера не найден.\n"
                "Запустите ячейки сохранения конфига в ноутбуке после обучения."
            )
            return

        # --- 4. Predict parameters ---------------------------------------
        try:
            reg_session = ort.InferenceSession(reg_onnx)
            params = predict_params(reg_session, self.dataset, self.k, reg_scaler)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка предсказания параметров", str(e))
            return

        if params is None:
            self._show_params("Ошибка: не удалось предсказать параметры.")
            return

        # --- 5. Display parameters ---------------------------------------
        self._show_params(format_params(params))

        # --- 6. Build and plot theoretical curve -------------------------
        t_D_array = self.dataset['t_D'].values
        t_D_array = t_D_array[t_D_array > 0]

        pred_df = build_predicted_curve(cls_name, params, t_D_array)
        self.generate_graph(pred_df)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _show_params(self, text):
        if hasattr(self, 'paramsEdit'):
            self.paramsEdit.setPlainText(text)

    def generate_graph(self, pred_df=None):
        if self.dataset is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для отображения!")
            return

        if not hasattr(self, 'plot_widget'):
            QMessageBox.warning(self, "Ошибка", "График не инициализирован!")
            return

        self.plot_widget.clear()

        required_keys = ['t_D', 'P_wD_gauss', 'dP_wD']
        missing = [k for k in required_keys if k not in self.dataset]
        if missing:
            QMessageBox.warning(self, "Ошибка", f"В датасете отсутствуют ключи: {missing}")
            return

        t = self.dataset['t_D']
        y = self.dataset['P_wD_gauss']
        dy = self.dataset['dP_wD']

        # Observed pressure (with noise)
        self.plot_widget.plot(
            t, y,
            pen=None,
            name='P_wD (наблюдение)',
            symbol='o',
            symbolSize=4,
            symbolBrush=pg.mkBrush('b'),
            symbolPen=None,
        )

        # Observed Bourdet derivative
        self.plot_widget.plot(
            t, dy,
            pen=pg.mkPen(color='b', width=2),
            name='dP/dlnt (наблюдение)',
        )

        # Predicted (theoretical) curve
        if pred_df is not None and len(pred_df) > 0:
            mask = (pred_df['t_D'] > 0) & (pred_df['P_wD'] > 0) & (pred_df['dP_wD'] > 0)
            pdf = pred_df[mask]
            if len(pdf) > 0:
                self.plot_widget.plot(
                    pdf['t_D'], pdf['P_wD'],
                    pen=pg.mkPen(color='r', width=2, style=Qt.DashLine),
                    name='P_wD (предсказание)',
                )
                self.plot_widget.plot(
                    pdf['t_D'], pdf['dP_wD'],
                    pen=pg.mkPen(color=(200, 0, 0), width=2),
                    name='dP/dlnt (предсказание)',
                )

        model_name = getattr(self, 'model_name', 'Предсказание')
        self.plot_widget.setTitle(f'График: {model_name}')
        self.plot_widget.setLabel('left', 'Значение')
        self.plot_widget.setLabel('bottom', 'Время')
        self.plot_widget.autoRange()

    def set_dataset(self, dataset, k, model_name=""):
        self.dataset = dataset
        self.k = k
        self.model_name = model_name

        self.generate_graph()

        if hasattr(self, 'lineEdit') and 'formation_type' in dataset:
            self.lineEdit.setText(str(dataset['formation_type']))
