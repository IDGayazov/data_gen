import os
import sys

from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)

import numpy as np
import pyqtgraph as pg

from model.reservoir_model import ReservoirModel
from generation.dual_permeability import InfiniteDualPermeabilityModelGenerator, FiniteDualPermeabilityGenerator
from generation.dual_porosity import InfiniteDualPorosityGenerator, FiniteDualPorosityGenerator
from generation.generator import GenerationParams, DataGenerator
from generation.homogeneous import InfiniteHomogeneousGenerator, FiniteHomogeneousGenerator
from generation.radial_composite import InfiniteRadialCompositeGenerator
from generation.utils import clear_folder

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class CommonReservoirModel(ReservoirModel):
    """
    Общая для всех реализация модели пласта

    Можно загрузить датасет из файла
    """
    def F(self, s):
        print('Not supported method!')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Получаем путь к .ui файлу
        import os
        ui_path = os.path.join(os.path.dirname(__file__), '/home/ilnaz/PycharmProjects/data-gen/ui/main.ui')
        
        # Загрузка UI
        uic.loadUi(ui_path, self)
        
        # Подключаем кнопку
        self.generate.clicked.connect(self.generate_graph)
        
        # Настраиваем график
        self.setup_loglog_plot()
    
    def setup_loglog_plot(self):
        """Настройка логарифмического графика"""
        # Создаем контейнер
        container = QWidget(self.centralwidget)
        container.setGeometry(self.graphic.geometry())
        
        # Layout для контейнера
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Создаем PlotWidget с логарифмическим масштабом
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        
        # Прячем старый QGraphicsView
        self.graphic.setVisible(False)
        
        # Включаем логарифмический режим для обеих осей
        self.plot_widget.setLogMode(x=True, y=True)
        
        # Настройка внешнего вида
        self.plot_widget.setLabel('left', 'Значение', units='')
        self.plot_widget.setLabel('bottom', 'Время', units='')
        self.plot_widget.setTitle('Логарифмический график (log-log)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setBackground('white')
        self.plot_widget.addLegend()
    
    def generate_graph(self):
        """Генерация графика"""

        file_name = self.get_single_file('/home/ilnaz/PycharmProjects/datasets/ui/curve')

        try:
            # Определяем выбранную модель
            if self.homogen_inf.isChecked():
                model_name = "Homogeneous infinite"

                self._make_homogeneous_inf_model()

                model = CommonReservoirModel()
                df = model.load_model(file_name) \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']
                color = 'b'
            elif self.homogen_fin.isChecked():
                model_name = "Homogeneous finite"

                self._make_homogeneous_fin_model()
                model = CommonReservoirModel()
                df = model.load_model('/home/ilnaz/PycharmProjects/datasets/ui/curve/homogeneous_fin_1.csv') \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']
                color = 'g'
            elif self.dual_por_inf.isChecked():
                model_name = "Dual porosity infinite"

                self._make_dual_porosity_inf_model()
                model = CommonReservoirModel()
                df = model.load_model('/home/ilnaz/PycharmProjects/datasets/ui/curve/dual_porosity_inf_1.csv') \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']
                color = 'r'
            elif self.dual_por_fin.isChecked():
                model_name = "Dual porosity finite"

                self._make_dual_porosity_fin_model()
                model = CommonReservoirModel()
                df = model.load_model('/home/ilnaz/PycharmProjects/datasets/ui/curve/dual_porosity_fin_1.csv') \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']
                color = 'c'
            elif self.dual_perm_inf.isChecked():
                model_name = "Dual permeability infinite"

                self._make_dual_permeability_inf_model()
                model = CommonReservoirModel()
                df = model.load_model('/home/ilnaz/PycharmProjects/datasets/ui/curve/dual_permeability_inf_1.csv') \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']
                color = 'm'
            elif self.dual_perm_fin.isChecked():
                model_name = "Dual permeability finite"

                self._make_dual_permeability_fin_model()
                model = CommonReservoirModel()
                df = model.load_model('/home/ilnaz/PycharmProjects/datasets/ui/curve/dual_permeability_fin_1.csv') \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']
                color = 'y'
            elif self.rad_comp_inf.isChecked():
                model_name = "Radial composite infinite"

                self._make_radial_composite_inf_model()
                model = CommonReservoirModel()
                df = model.load_model('/home/ilnaz/PycharmProjects/datasets/ui/curve/radial_composite_inf_1.csv') \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']
                color = '#FF6B6B'
            else:
                QMessageBox.warning(self, "Предупреждение", "Пожалуйста, выберите тип модели!")
                return
            
            # Очищаем график
            self.plot_widget.clear()
            
            # Строим график
            self.plot_widget.plot(
                t, y,
                pen=pg.mkPen(color=color, width=3),
                name='Pressure',
                symbol='o',
                symbolSize=5,
                symbolBrush=color
            )

            self.plot_widget.plot(
                t, dy,
                pen=pg.mkPen(color=color, width=3),
                name='Pressure derivative',
                symbol='o',
                symbolSize=5,
                symbolBrush='r'
            )
            
            # Настройка
            self.plot_widget.setTitle(f'График: {model_name}')
            self.plot_widget.setLabel('left', 'Значение')
            self.plot_widget.setLabel('bottom', 'Время')
            self.plot_widget.autoRange()
            
            QMessageBox.information(self, "Успех", f"График для '{model_name}' сгенерирован!")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка: {str(e)}")

    def get_single_file(self, folder_path):
        """Возвращает путь к единственному файлу в папке"""
        files = os.listdir(folder_path)
        
        if len(files) == 0:
            raise ValueError(f"В папке {folder_path} нет файлов")
        elif len(files) == 1:
            return os.path.join(folder_path, files[0])
        else:
            raise ValueError(f"В папке {folder_path} найдено {len(files)} файлов, ожидался один")

    def _make_homogeneous_inf_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-5
        OUTPUT_PATH: Final = '../datasets/ui'

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = InfiniteHomogeneousGenerator(params)
        generator.generate()

    def _make_homogeneous_fin_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = '../datasets/ui'

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = FiniteHomogeneousGenerator(params)
        generator.generate()

    def _make_dual_porosity_inf_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = '../datasets/ui'

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = InfiniteDualPorosityGenerator(params)
        generator.generate()

    def _make_dual_porosity_fin_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = '../datasets/ui'

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = FiniteDualPorosityGenerator(params)
        generator.generate()

    def _make_dual_permeability_inf_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = '../datasets/ui'

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = InfiniteDualPermeabilityModelGenerator(params)
        generator.generate()

    def _make_dual_permeability_fin_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = '../datasets/ui'

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = FiniteDualPermeabilityGenerator(params)
        generator.generate()

    def _make_radial_composite_inf_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = '../datasets/ui'

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = InfiniteRadialCompositeGenerator(params)
        generator.generate()

# Точка входа
if __name__ == '__main__':
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())