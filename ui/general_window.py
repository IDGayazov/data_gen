import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from pathlib import Path

# TF must be imported before PyQt5 to avoid DLL conflicts on Windows
from tensorflow import keras  # noqa: F401

import PyQt5
from PyQt5.QtCore import QLibraryInfo

import pandas as pd

# On some Windows setups Qt fails to auto-discover plugins in venv paths.
_qt_plugins_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
_qt_plugins_dir = Path(_qt_plugins_path) if _qt_plugins_path else None
if not (_qt_plugins_dir and _qt_plugins_dir.exists()):
    # Fallback for environments where QLibraryInfo returns an empty path.
    _pyqt5_file = getattr(PyQt5, '__file__', None)
    if _pyqt5_file:
        _candidate = Path(_pyqt5_file).resolve().parent / 'Qt5' / 'plugins'
        if _candidate.exists():
            _qt_plugins_dir = _candidate
if not (_qt_plugins_dir and _qt_plugins_dir.exists()):
    _candidate = Path(sys.prefix) / 'Lib' / 'site-packages' / 'PyQt5' / 'Qt5' / 'plugins'
    if _candidate.exists():
        _qt_plugins_dir = _candidate

if _qt_plugins_dir and _qt_plugins_dir.exists():
    _qt_platforms_dir = _qt_plugins_dir / 'platforms'
    os.environ.setdefault('QT_PLUGIN_PATH', str(_qt_plugins_dir))
    if _qt_platforms_dir.exists():
        os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', str(_qt_platforms_dir))

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

from generation.show_generated_model import CommonReservoirModel
from ui.prediction import GenerationWindow 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Получаем путь к .ui файлу
        import os
        ui_path = os.path.join(os.path.dirname(__file__), 'main.ui')
        
        # Загрузка UI
        uic.loadUi(ui_path, self)
        
        # Подключаем кнопку
        self.generate.clicked.connect(self.generate_graph)
        
        # Настраиваем график
        self.setup_loglog_plot()
        self.predict.clicked.connect(self.open_generation_window)

        # Ссылка на окно предсказания
        self.prediction_window = None
        self.df = None
        self.k = 1

    def open_generation_window(self):
        """Открывает окно prediction.ui"""
        try:
            # Создаем окно, если его еще нет или оно закрыто
            if self.prediction_window is None or not self.prediction_window.isVisible():
                self.prediction_window = GenerationWindow(self)
                self.prediction_window.set_dataset(self.df, self.k)
                self.prediction_window.show()
            else:
                # Если окно уже открыто, просто активируем его
                self.prediction_window.raise_()
                self.prediction_window.activateWindow()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть окно предсказания: {str(e)}")

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

        try:
            # Определяем выбранную модель
            if self.homogen_inf.isChecked():
                model_name = "Homogeneous infinite"

                self._make_homogeneous_inf_model()

                file_name = self.get_single_file(str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui' / 'curve'))
                file_name_params = self.get_single_file(str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui' / 'params'))

                params_df = pd.read_csv(file_name_params)
                self.k = params_df['k']

                model = CommonReservoirModel()
                df = model.load_model(file_name) \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']

                self.df = df

                color = 'b'
            elif self.homogen_fin.isChecked():
                model_name = "Homogeneous finite"

                self._make_homogeneous_fin_model()

                file_name = self.get_single_file(str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui' / 'curve'))

                model = CommonReservoirModel()
                df = model.load_model(file_name) \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']

                self.df = df

                color = 'g'
            elif self.dual_por_inf.isChecked():
                model_name = "Dual porosity infinite"

                self._make_dual_porosity_inf_model()

                file_name = self.get_single_file(str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui' / 'curve'))

                model = CommonReservoirModel()
                df = model.load_model(file_name) \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']

                self.df = df

                color = 'r'
            elif self.dual_por_fin.isChecked():
                model_name = "Dual porosity finite"

                self._make_dual_porosity_fin_model()
                
                file_name = self.get_single_file(str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui' / 'curve'))

                model = CommonReservoirModel()
                df = model.load_model(file_name) \
                          .get_pressure()

                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']

                self.df = df

                color = 'c'
            elif self.dual_perm_inf.isChecked():
                model_name = "Dual permeability infinite"

                self._make_dual_permeability_inf_model()

                file_name = self.get_single_file(str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui' / 'curve'))

                model = CommonReservoirModel()
                df = model.load_model(file_name) \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']

                self.df = df

                color = 'm'
            elif self.dual_perm_fin.isChecked():
                model_name = "Dual permeability finite"

                self._make_dual_permeability_fin_model()

                file_name = self.get_single_file(str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui' / 'curve'))

                model = CommonReservoirModel()
                df = model.load_model(file_name) \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']

                self.df = df

                color = 'y'
            elif self.rad_comp_inf.isChecked():
                model_name = "Radial composite infinite"

                self._make_radial_composite_inf_model()

                file_name = self.get_single_file(str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui' / 'curve'))

                model = CommonReservoirModel()
                df = model.load_model(file_name) \
                          .get_pressure()
                
                t = df['t_D']
                y = df['P_wD_gauss']
                dy = df['dP_wD']

                self.df = df

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
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Папка не найдена: {folder_path}")
        files = [f for f in os.listdir(folder_path) if not f.startswith('.')]

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
        OUTPUT_PATH: Final = str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui')

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = InfiniteHomogeneousGenerator(params)
        generator.generate()

    def _make_homogeneous_fin_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui')

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = FiniteHomogeneousGenerator(params)
        generator.generate()

    def _make_dual_porosity_inf_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui')

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = InfiniteDualPorosityGenerator(params)
        generator.generate()

    def _make_dual_porosity_fin_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui')

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = FiniteDualPorosityGenerator(params)
        generator.generate()

    def _make_dual_permeability_inf_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui')

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = InfiniteDualPermeabilityModelGenerator(params)
        generator.generate()

    def _make_dual_permeability_fin_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui')

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = FiniteDualPermeabilityGenerator(params)
        generator.generate()

    def _make_radial_composite_inf_model(self):
        T_MAX_DAYS: Final = 30
        POINTS_COUNT: Final = 128
        SIZE: Final = 1
        SIGMA: Final = 5e-3
        OUTPUT_PATH: Final = str(Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ui')

        clear_folder(OUTPUT_PATH)

        params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

        generator = InfiniteRadialCompositeGenerator(params)
        generator.generate()

# Точка входа
if __name__ == '__main__':
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())