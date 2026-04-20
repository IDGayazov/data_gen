import os
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
import re
from typing import Dict, List, Tuple

class FileStatisticsAnalyzer:
    def __init__(self):
        self.data = []
        self.statistics = {}
        
        # Определяем известные составные типы пласта (в порядке убывания длины)
        self.reservoir_types = [
            'dual_permeability',
            'dual_porosity', 
            'radial_composite',
            'homogeneous'
        ]
        
        # Типы границ
        self.boundary_types = [
            'permeability',
            'porosity',
            'fin',
            'inf',
            'composite'
        ]
    
    def parse_filename(self, filename: str) -> Dict:
        """
        Парсинг имени файла.
        Формат: тип_пласта_граница_номер_суффикс.csv
        Например: dual_permeability_permeability_3326_full.csv
                 dual_porosity_porosity_3333_inc1.csv
                 radial_composite_composite_1663_inc2.csv
                 homogeneous_fin_1688_full.csv
        """
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]
        
        # Определяем суффикс (inc1, inc2, full)
        suffix = None
        for possible_suffix in ['inc1', 'inc2', 'full']:
            if name_without_ext.endswith(f'_{possible_suffix}'):
                suffix = possible_suffix
                # Удаляем суффикс из имени
                name_without_ext = name_without_ext.replace(f'_{suffix}', '')
                break
        
        # Теперь парсим основную часть имени
        parts = name_without_ext.split('_')
        
        result = {
            'filename': basename,
            'full_path': filename,
            'suffix': suffix,
            'reservoir_type': None,
            'boundary_type': None,
            'number': None
        }
        
        # Определяем тип пласта (пробуем найти среди составных типов)
        # Собираем имя постепенно, начиная с первого слова
        for i in range(len(parts), 0, -1):
            candidate = '_'.join(parts[:i])
            if candidate in self.reservoir_types:
                result['reservoir_type'] = candidate
                remaining_parts = parts[i:]
                break
        else:
            # Если не нашли среди известных, берем первую часть
            result['reservoir_type'] = parts[0] if parts else 'unknown'
            remaining_parts = parts[1:]
        
        # Определяем тип границы из оставшихся частей
        if remaining_parts:
            # Пробуем найти границу
            for boundary in self.boundary_types:
                if remaining_parts[0] == boundary:
                    result['boundary_type'] = boundary
                    remaining_parts = remaining_parts[1:]
                    break
            else:
                # Если не нашли, возможно это номер
                result['boundary_type'] = remaining_parts[0] if remaining_parts else None
                remaining_parts = remaining_parts[1:] if remaining_parts else []
        
        # Определяем номер (должен быть цифрой)
        if remaining_parts and remaining_parts[0].isdigit():
            result['number'] = remaining_parts[0]
        
        return result
    
    def scan_directory(self, directory: str, pattern: str = "*.csv") -> List[Dict]:
        """
        Сканирование директории и парсинг всех файлов
        """
        files = glob.glob(os.path.join(directory, pattern))
        
        print(f"Найдено файлов: {len(files)}")
        
        for file in files:
            parsed = self.parse_filename(file)
            self.data.append(parsed)
        
        return self.data
    
    def create_dataframe(self) -> pd.DataFrame:
        """
        Создание DataFrame из собранных данных
        """
        self.df = pd.DataFrame(self.data)
        return self.df
    
    def group_statistics(self) -> Dict:
        """
        Группированная статистика по типам пласта, границам и суффиксам
        """
        if not hasattr(self, 'df') or self.df.empty:
            print("Нет данных. Сначала выполните scan_directory()")
            return {}
        
        # Группировка: тип пласта + граница + суффикс
        grouped = self.df.groupby(['reservoir_type', 'boundary_type', 'suffix']).size().reset_index()
        grouped.columns = ['reservoir_type', 'boundary_type', 'suffix', 'count']
        
        # Создаем сводную таблицу
        pivot = grouped.pivot_table(
            index=['reservoir_type', 'boundary_type'],
            columns='suffix',
            values='count',
            fill_value=0
        )
        
        # Добавляем итоговую строку
        total_row = pivot.sum()
        pivot.loc[('Total', 'Total')] = total_row
        
        # Добавляем процентное распределение
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        
        self.statistics = {
            'counts': pivot,
            'percentages': pivot_pct,
            'detailed': grouped
        }
        
        return self.statistics
    
    def print_statistics(self):
        """
        Вывод статистики в читаемом формате
        """
        if not self.statistics:
            print("Статистика не рассчитана. Выполните group_statistics()")
            return
        
        print("\n" + "="*90)
        print("ГРУППИРОВАННАЯ СТАТИСТИКА ПО ТИПАМ ПЛАСТА, ГРАНИЦАМ И СУФФИКСАМ")
        print("="*90)
        
        # Абсолютные значения
        print("\n📊 АБСОЛЮТНЫЕ ЗНАЧЕНИЯ (количество файлов):")
        print("-"*90)
        print(self.statistics['counts'].to_string())
        
        # Проценты
        print("\n📈 ПРОЦЕНТНОЕ РАСПРЕДЕЛЕНИЕ (%):")
        print("-"*90)
        print(self.statistics['percentages'].round(2).to_string())
        
        # Детальная статистика
        print("\n🔍 ДЕТАЛЬНАЯ СТАТИСТИКА:")
        print("-"*90)
        print(f"{'Тип пласта':<25} | {'Граница':<15} | {'inc1':>6} | {'inc2':>6} | {'full':>6} | {'Total':>6}")
        print("-"*90)
        
        for idx in self.statistics['counts'].index:
            if idx[0] != 'Total':
                reservoir, boundary = idx
                row = self.statistics['counts'].loc[idx]
                total = row.sum()
                print(f"{reservoir:<25} | {boundary:<15} | {row.get('inc1', 0):>6} | "
                      f"{row.get('inc2', 0):>6} | {row.get('full', 0):>6} | {total:>6}")
        
        # Итоговая строка
        print("-"*90)
        total_row = self.statistics['counts'].loc[('Total', 'Total')]
        print(f"{'ВСЕГО':<25} | {'':15} | {total_row.get('inc1', 0):>6} | "
              f"{total_row.get('inc2', 0):>6} | {total_row.get('full', 0):>6} | {total_row.sum():>6}")
    
    def print_grouped_by_reservoir(self):
        """
        Вывод статистики, сгруппированной по типам пласта
        """
        if not hasattr(self, 'df') or self.df.empty:
            print("Нет данных. Сначала выполните scan_directory()")
            return
        
        print("\n" + "="*80)
        print("СТАТИСТИКА ПО ТИПАМ ПЛАСТА (суммарно по всем границам)")
        print("="*80)
        
        # Группируем по типу пласта и суффиксу
        grouped = self.df.groupby(['reservoir_type', 'suffix']).size().unstack(fill_value=0)
        
        print("\n📊 АБСОЛЮТНЫЕ ЗНАЧЕНИЯ:")
        print(grouped.to_string())
        
        print("\n📈 ПРОЦЕНТНОЕ РАСПРЕДЕЛЕНИЕ (%):")
        grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
        print(grouped_pct.round(2).to_string())
        
        return grouped
    
    def save_statistics(self, output_dir: str):
        """
        Сохранение статистики в файлы
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем сводную таблицу
        self.statistics['counts'].to_csv(
            os.path.join(output_dir, 'statistics_counts.csv')
        )
        
        # Сохраняем проценты
        self.statistics['percentages'].to_csv(
            os.path.join(output_dir, 'statistics_percentages.csv')
        )
        
        # Сохраняем детальную статистику
        self.statistics['detailed'].to_csv(
            os.path.join(output_dir, 'statistics_detailed.csv'),
            index=False
        )
        
        # Сохраняем полный DataFrame
        self.df.to_csv(os.path.join(output_dir, 'all_files_parsed.csv'), index=False)
        
        # Сохраняем статистику по типам пласта
        reservoir_stats = self.df.groupby(['reservoir_type', 'suffix']).size().unstack(fill_value=0)
        reservoir_stats.to_csv(os.path.join(output_dir, 'statistics_by_reservoir.csv'))
        
        print(f"\n💾 Статистика сохранена в: {output_dir}")
    
    def check_balance(self) -> pd.DataFrame:
        """
        Проверка баланса между группами inc1, inc2, full для каждого типа пласта и границы
        """
        if not hasattr(self, 'df') or self.df.empty:
            print("Нет данных. Сначала выполните scan_directory()")
            return pd.DataFrame()
        
        # Группировка по типу пласта, границе и суффиксу
        grouped = self.df.groupby(['reservoir_type', 'boundary_type', 'suffix']).size().unstack(fill_value=0)
        
        # Добавляем проверку на баланс
        balance_df = grouped.copy()
        
        for col in ['inc1', 'inc2', 'full']:
            if col in balance_df.columns:
                balance_df[f'{col}_is_balanced'] = balance_df[col] >= 10
        
        balance_df['overall_balanced'] = balance_df[[c for c in balance_df.columns if c.endswith('_is_balanced')]].all(axis=1)
        
        print("\n⚖️ ПРОВЕРКА БАЛАНСА ГРУПП:")
        print("-"*80)
        print(balance_df.to_string())
        
        return balance_df


def main_analysis():
    # Путь к директории с обработанными файлами
    PROCESSED_DIR = "/home/ilnaz/PycharmProjects/datasets/models_incs_prc/curve"
    OUTPUT_DIR = "/home/ilnaz/PycharmProjects/datasets/models_incs_prc/curve/statistics"
    
    print("="*80)
    print("АНАЛИЗ СТАТИСТИКИ ФАЙЛОВ")
    print("="*80)
    
    analyzer = FileStatisticsAnalyzer()
    
    # Сканирование директории
    data = analyzer.scan_directory(PROCESSED_DIR, pattern="*.csv")
    df = analyzer.create_dataframe()
    
    print(f"\nУникальные типы пласта в данных: {df['reservoir_type'].unique().tolist()}")
    print(f"Уникальные типы границ: {df['boundary_type'].unique().tolist()}")
    print(f"Уникальные суффиксы: {df['suffix'].unique().tolist()}")
    
    print("\nПервые 10 записей:")
    print(df.head(10))
    
    # Расчет статистики
    stats = analyzer.group_statistics()
    analyzer.print_statistics()
    
    # Статистика по типам пласта
    reservoir_stats = analyzer.print_grouped_by_reservoir()
    
    # Проверка баланса
    balance = analyzer.check_balance()
    
    # Сохранение результатов
    analyzer.save_statistics(OUTPUT_DIR)
    
    # Вывод сводной информации
    print("\n" + "="*80)
    print("СВОДНАЯ ИНФОРМАЦИЯ")
    print("="*80)
    
    total_files = len(df)
    print(f"\n📁 Всего файлов: {total_files}")
    
    print(f"\n📊 Распределение по суффиксам:")
    for suffix, count in df['suffix'].value_counts().items():
        pct = count / total_files * 100
        print(f"   {suffix}: {count} ({pct:.1f}%)")
    
    print(f"\n📊 Распределение по типам пласта:")
    for reservoir, count in df['reservoir_type'].value_counts().items():
        pct = count / total_files * 100
        print(f"   {reservoir}: {count} ({pct:.1f}%)")
    
    print(f"\n📊 Распределение по границам:")
    for boundary, count in df['boundary_type'].value_counts().items():
        pct = count / total_files * 100
        print(f"   {boundary}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main_analysis()