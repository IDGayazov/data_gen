import os
import sys
import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QWidget, QVBoxLayout, QMessageBox
from PyQt5.QtCore import Qt

from ui.load_model import load_cls_model, predict


class GenerationWindow(QDialog):
    def __init__(self, dataset=None, parent=None):
        super().__init__(parent)
        # Исправлен путь к UI файлу (без абсолютного пути)
        ui_path = os.path.join(os.path.dirname(__file__), 'prediction.ui')
        uic.loadUi(ui_path, self)

        self.dataset = dataset
        self.model_name = ""  # Добавляем атрибут для имени модели

        # Подключаем кнопку (убедитесь, что в prediction.ui есть кнопка с именем prediction)
        if hasattr(self, 'prediction'):
            self.prediction.clicked.connect(self.predict)

        # Настраиваем график
        self.setup_loglog_plot()

    def setup_loglog_plot(self):
        """Настройка логарифмического графика"""
        # Проверяем существование graphicsView
        if not hasattr(self, 'graphicsView'):
            QMessageBox.warning(self, "Ошибка", "В UI файле отсутствует graphicsView!")
            return
            
        # Создаем контейнер, который будет замещать graphicsView
        container = QWidget(self)
        container.setGeometry(self.graphicsView.geometry())
        
        # Layout для контейнера
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Создаем PlotWidget с логарифмическим масштабом
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        
        # Прячем старый QGraphicsView
        self.graphicsView.setVisible(False)
        
        # Показываем контейнер с новым графиком
        container.show()
        
        # Включаем логарифмический режим для обеих осей
        self.plot_widget.setLogMode(x=True, y=True)
        
        # Настройка внешнего вида
        self.plot_widget.setLabel('left', 'Значение', units='')
        self.plot_widget.setLabel('bottom', 'Время', units='')
        self.plot_widget.setTitle('Логарифмический график (log-log)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setBackground('white')
        self.plot_widget.addLegend()
        
        # Сохраняем ссылку на контейнер для возможного использования
        self.plot_container = container

    def predict(self):
        cls_model = ""
        if self.cnn.isChecked():
            cls_model = '1d_cnn'
        elif self.cnn_lstm.isChecked():
            cls_model = 'cnn_lstm'
        else:
            cls_model = 'lstm'

        model = load_cls_model(cls_model)
        cls_name = predict(model, self.dataset, self.k)
        self.lineEdit.setText(cls_name)

    def generate_graph(self):
        """Генерация графика"""
        try:
            # Проверяем наличие датасета
            if self.dataset is None:
                QMessageBox.warning(self, "Предупреждение", "Нет данных для отображения!")
                return
            
            # Проверяем наличие plot_widget
            if not hasattr(self, 'plot_widget'):
                QMessageBox.warning(self, "Ошибка", "График не инициализирован!")
                return
            
            # Очищаем график
            self.plot_widget.clear()

            # Проверяем наличие необходимых ключей в датасете
            required_keys = ['t_D', 'P_wD_gauss', 'dP_wD']
            missing_keys = [key for key in required_keys if key not in self.dataset]
            if missing_keys:
                QMessageBox.warning(self, "Ошибка", f"В датасете отсутствуют ключи: {missing_keys}")
                return
            
            # Получаем данные
            t = self.dataset['t_D']
            y = self.dataset['P_wD_gauss']
            dy = self.dataset['dP_wD']
            
            # Определяем цвет (можно сделать параметром)
            color = 'b'  # синий цвет по умолчанию
            
            # Строим график давления
            self.plot_widget.plot(
                t, y,
                pen=pg.mkPen(color=color, width=3),
                name='Pressure',
                symbol='o',
                symbolSize=5,
                symbolBrush=color
            )
            
            # Строим график производной
            self.plot_widget.plot(
                t, dy,
                pen=pg.mkPen(color='r', width=3),
                name='Pressure derivative',
                symbol='o',
                symbolSize=5,
                symbolBrush='r'
            )
            
            # Настройка графика
            model_name = getattr(self, 'model_name', 'Предсказание')
            self.plot_widget.setTitle(f'График: {model_name}')
            self.plot_widget.setLabel('left', 'Значение')
            self.plot_widget.setLabel('bottom', 'Время')
            self.plot_widget.autoRange()
            
            QMessageBox.information(self, "Успех", "График успешно сгенерирован!")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при генерации графика: {str(e)}")
    
    def set_dataset(self, dataset, k, model_name=""):
        """Метод для установки датасета после создания окна"""
        self.dataset = dataset
        self.k = k
        self.model_name = model_name
        # Автоматически генерируем график
        self.generate_graph()
        
        # Если есть поле lineEdit, обновляем его с типом пласта
        if hasattr(self, 'lineEdit') and 'formation_type' in dataset:
            self.lineEdit.setText(str(dataset['formation_type']))
    
    def resizeEvent(self, event):
        """Обработка изменения размера окна"""
        super().resizeEvent(event)
        # Обновляем размер контейнера с графиком при изменении размера окна
        if hasattr(self, 'plot_container') and hasattr(self, 'graphicsView'):
            self.plot_container.setGeometry(self.graphicsView.geometry())