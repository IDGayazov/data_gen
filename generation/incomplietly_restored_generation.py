import os
import pandas as pd
import numpy as np
from scipy import interpolate
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import random

class ThreeGroupWellTestProcessor:
    def __init__(self):
        self.processed_files = []
        self.error_files = []
        self.group_assignment = {}
        
    def scan_directory(self, directory: str, extension: str = '.csv') -> List[str]:
        """
        Сканирование директории и поиск всех файлов
        """
        pattern = os.path.join(directory, f'*{extension}')
        files = glob.glob(pattern)
        print(f"Найдено файлов: {len(files)} в директории {directory}")
        return files
    
    def assign_groups(self, files: List[str], seed: int = 42) -> Dict[str, List[str]]:
        """
        Случайное распределение файлов по трем группам
        
        Returns:
        --------
        Dict: {'inc1': [...], 'inc2': [...], 'full': [...]}
        """
        # Перемешиваем файлы
        random.seed(seed)
        files_shuffled = files.copy()
        random.shuffle(files_shuffled)
        
        # Разделение на 3 группы
        n_files = len(files_shuffled)
        n_per_group = n_files // 3
        
        groups = {
            'inc1': files_shuffled[:n_per_group],           # Короткие (0-1e3)
            'inc2': files_shuffled[n_per_group:2*n_per_group], # Средние (0-1e6)
            'full': files_shuffled[2*n_per_group:]           # Полные (без обрезания)
        }
        
        # Статистика
        print(f"\n{'='*50}")
        print(f"РАСПРЕДЕЛЕНИЕ КРИВЫХ ПО ГРУППАМ:")
        print(f"  inc1 (0-1e3):   {len(groups['inc1'])} кривых")
        print(f"  inc2 (0-1e6):   {len(groups['inc2'])} кривых")
        print(f"  full (полные):  {len(groups['full'])} кривых")
        print(f"{'='*50}\n")
        
        # Сохраняем назначение для каждого файла
        for file in files_shuffled:
            if file in groups['inc1']:
                self.group_assignment[file] = 'inc1'
            elif file in groups['inc2']:
                self.group_assignment[file] = 'inc2'
            else:
                self.group_assignment[file] = 'full'
        
        return groups
    
    def preprocess_curve(self, df: pd.DataFrame, n_points: int = 128,
                        t_max_crop: Optional[float] = None,
                        normalize: bool = True) -> pd.DataFrame:
        """
        Модифицированная предобработка кривой с обрезанием и 
        БЕЗОПАСНОЙ линейной интерполяцией в лог-пространстве.
        """
        required_cols = ['t_D', 'dP_wD']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame должен содержать колонку '{col}'")
        
        has_gauss = 'P_wD_gauss' in df.columns

        df = df.sort_values('t_D').reset_index(drop=True)
        df = df[(df['t_D'] > 0) & (df['dP_wD'] > 0)].reset_index(drop=True)
        
        if t_max_crop is not None:
            df = df[df['t_D'] <= t_max_crop].reset_index(drop=True)
        
        df = df.drop_duplicates(subset=['t_D']).reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError(f"После обрезания и фильтрации не осталось данных (t_max_crop={t_max_crop})")
        
        if len(df) > 1:
            log_t_min = np.log10(df['t_D'].min())
            log_t_max = np.log10(df['t_D'].max())
            
            if log_t_max > log_t_min:
                log_t_interp = np.linspace(log_t_min, log_t_max, n_points)
                t_interp = 10 ** log_t_interp
                                log_dP_values = np.log10(df['dP_wD'].values)
                
                f_dP_log = interpolate.interp1d(np.log10(df['t_D'].values), 
                                               log_dP_values,
                                               kind='linear',
                                               bounds_error=False,
                                               fill_value='extrapolate')
                
                dP_interp = 10 ** f_dP_log(log_t_interp)
                
                if has_gauss:
                    if (df['P_wD_gauss'] > 0).all():
                        f_gauss_log = interpolate.interp1d(np.log10(df['t_D'].values), 
                                                           np.log10(df['P_wD_gauss'].values),
                                                           kind='linear',
                                                           bounds_error=False,
                                                           fill_value='extrapolate')
                        gauss_interp = 10 ** f_gauss_log(log_t_interp)
                    else:
                        f_gauss = interpolate.interp1d(df['t_D'].values, 
                                                      df['P_wD_gauss'].values,
                                                      kind='linear',
                                                      bounds_error=False,
                                                      fill_value='extrapolate')
                        gauss_interp = f_gauss(t_interp)
                else:
                    gauss_interp = np.zeros(n_points)
            else:
                t_interp = np.linspace(df['t_D'].min(), df['t_D'].max(), n_points)
                dP_interp = np.full(n_points, df['dP_wD'].mean())
                gauss_interp = np.full(n_points, df['P_wD_gauss'].mean()) if has_gauss else np.zeros(n_points)
        else:
            t_interp = np.full(n_points, df['t_D'].iloc[0])
            dP_interp = np.full(n_points, df['dP_wD'].iloc[0])
            gauss_interp = np.full(n_points, df['P_wD_gauss'].iloc[0]) if has_gauss else np.zeros(n_points)
        
        result_df = pd.DataFrame({
            't_D': t_interp,
            'dP_wD': dP_interp
        })
        
        if has_gauss:
            result_df['P_wD_gauss'] = gauss_interp
        
        if normalize:
            t_min, t_max = result_df['t_D'].min(), result_df['t_D'].max()
            if t_max > t_min:
                result_df['t_D_norm'] = (result_df['t_D'] - t_min) / (t_max - t_min)
            else:
                result_df['t_D_norm'] = 0.5
            
            p_min, p_max = result_df['dP_wD'].min(), result_df['dP_wD'].max()
            if p_max > p_min:
                result_df['dP_wD_norm'] = (result_df['dP_wD'] - p_min) / (p_max - p_min)
            else:
                result_df['dP_wD_norm'] = 0.5
            
            if has_gauss:
                g_min, g_max = result_df['P_wD_gauss'].min(), result_df['P_wD_gauss'].max()
                if g_max > g_min:
                    result_df['P_wD_gauss_norm'] = (result_df['P_wD_gauss'] - g_min) / (g_max - g_min)
                else:
                    result_df['P_wD_gauss_norm'] = 0.5
        
        return result_df
    
    def _compute_log_cut(self, df: pd.DataFrame, fraction: Optional[float]) -> Optional[float]:
        """
        Вычисляет t_max_crop как fraction долю лог-диапазона кривой.
        fraction=None → без обрезания (full).
        """
        if fraction is None:
            return None
        t_vals = df['t_D'].dropna()
        t_vals = t_vals[t_vals > 0]
        if len(t_vals) < 2:
            return None
        log_min = np.log10(t_vals.min())
        log_max = np.log10(t_vals.max())
        return 10 ** (log_min + fraction * (log_max - log_min))

    def process_single_file(self, args: Tuple) -> Dict:
        """
        Обработка одного файла в соответствии с его группой
        """
        input_file, output_dir, n_points, group_config, normalize, overwrite = args

        try:
            df = pd.read_csv(input_file)
            base_name = os.path.basename(input_file)
            name_without_ext = os.path.splitext(base_name)[0]


            group = self.group_assignment.get(input_file, 'full')

            fraction = group_config[group].get('fraction')  
            t_max_crop = self._compute_log_cut(df, fraction)
            suffix = group_config[group]['suffix']

            df_processed = self.preprocess_curve(
                df,
                n_points=n_points,
                t_max_crop=t_max_crop,
                normalize=normalize
            )
            
            output_filename = f"{name_without_ext}_{suffix}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            if os.path.exists(output_path) and not overwrite:
                return {
                    'file': input_file,
                    'name': name_without_ext,
                    'group': group,
                    'status': 'skipped',
                    'output': output_path
                }
            
            df_processed.to_csv(output_path, index=False)
            
            return {
                'file': input_file,
                'name': name_without_ext,
                'group': group,
                'status': 'success',
                'output': output_path,
                't_max_crop': t_max_crop,
                'n_points': len(df_processed),
                'has_gauss': 'P_wD_gauss' in df_processed.columns
            }
            
        except Exception as e:
            return {
                'file': input_file,
                'status': 'error',
                'error': str(e)
            }
    
    def process_all_files(self, input_dir: str, output_dir: str,
                         n_points: int = 128,
                         group_config: Dict = None,
                         normalize: bool = True,
                         overwrite: bool = True,
                         parallel: bool = True,
                         n_workers: int = 4,
                         file_extension: str = '.csv',
                         random_seed: int = 42):
        """
        Обработка всех файлов с распределением по трем группам
        """
        if group_config is None:
            group_config = {
                'inc1': {'fraction': 1/3, 'suffix': 'inc1', 'description': 'первая треть лог-диапазона'},
                'inc2': {'fraction': 2/3, 'suffix': 'inc2', 'description': 'первые две трети лог-диапазона'},
                'full': {'fraction': None, 'suffix': 'full', 'description': 'полные (без обрезания)'}
            }
        
        all_files = self.scan_directory(input_dir, extension=file_extension)
        
        if len(all_files) == 0:
            print(f"Не найдено файлов в {input_dir}")
            return
        
        groups = self.assign_groups(all_files, seed=random_seed)
        
        os.makedirs(output_dir, exist_ok=True)
        
        args_list = [(f, output_dir, n_points, group_config, normalize, overwrite) 
                     for f in all_files]
        
        if parallel:
            results = self._process_parallel(args_list, n_workers)
        else:
            results = self._process_sequential(args_list)
        
        self._analyze_results(results, group_config)
        
        self._create_summary(output_dir, results, group_config)
        
        return results
    
    def _process_parallel(self, args_list: List, n_workers: int) -> List:
        """Параллельная обработка файлов"""
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(self.process_single_file, args): args[0] 
                      for args in args_list}
            
            with tqdm(total=len(futures), desc="Обработка файлов") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
        return results
    
    def _process_sequential(self, args_list: List) -> List:
        """Последовательная обработка файлов"""
        results = []
        for args in tqdm(args_list, desc="Обработка файлов"):
            result = self.process_single_file(args)
            results.append(result)
        return results
    
    def _analyze_results(self, results: List, group_config: Dict):
        """Анализ и вывод статистики"""
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] == 'error')
        skipped_count = sum(1 for r in results if r['status'] == 'skipped')
        
        group_stats = {group: {'success': 0, 'skipped': 0, 'error': 0} 
                      for group in group_config.keys()}
        
        for result in results:
            if result['status'] == 'success':
                group_stats[result['group']]['success'] += 1
            elif result['status'] == 'skipped':
                group_stats[result['group']]['skipped'] += 1
            elif result['status'] == 'error':
                file_path = result['file']
                if file_path in self.group_assignment:
                    group = self.group_assignment[file_path]
                    group_stats[group]['error'] += 1
                else:
                    print(f"Предупреждение: файл {file_path} не найден в распределении групп")
        
        print(f"\n{'='*60}")
        print(f"ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"{'='*60}")
        print(f"Всего файлов: {len(results)}")
        print(f"Успешно: {success_count}")
        print(f"Пропущено: {skipped_count}")
        print(f"Ошибок: {error_count}")
        
        print(f"\nСтатистика по группам:")
        for group, stats in group_stats.items():
            desc = group_config[group]['description']
            print(f"  {group} ({desc}):")
            print(f"    - успешно: {stats['success']}")
            print(f"    - пропущено: {stats['skipped']}")
            print(f"    - ошибок: {stats['error']}")

    def _create_summary(self, output_dir: str, results: List, group_config: Dict):
        """Сохранение CSV-сводки результатов обработки"""
        summary_data = []
        for r in results:
            row = {
                'file': os.path.basename(r['file']),
                'group': r.get('group', ''),
                'status': r['status'],
                'output': r.get('output', ''),
                'error': r.get('error', ''),
            }
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'processing_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Сводка сохранена: {summary_path}")

    def save_group_assignment(self, output_dir: str):
        """Сохранение информации о распределении по группам"""
        assignment_data = []
        for file, group in self.group_assignment.items():
            assignment_data.append({
                'file': os.path.basename(file),
                'full_path': file,
                'group': group
            })

        assignment_df = pd.DataFrame(assignment_data)
        assignment_path = os.path.join(output_dir, 'group_assignment.csv')
        assignment_df.to_csv(assignment_path, index=False)
        print(f"Распределение по группам сохранено: {assignment_path}")


def main():
    # Конфигурация
    CONFIG = {
        'input_dir': r'C:\Users\gayaz\OneDrive\Рабочий стол\Code\диплом\datasets\cls_data\cls_data\curve',           # Папка с исходными кривыми
        'output_dir': r'C:\Users\gayaz\OneDrive\Рабочий стол\Code\диплом\datasets\cls_incs_data\curve',    # Папка для обработанных кривых
        'n_points': 128,                     # Количество точек после интерполяции
        'normalize': True,                   # Нормализация данных
        'overwrite': True,                   # Перезаписывать существующие файлы
        'parallel': True,                    # Параллельная обработка
        'n_workers': 8,                      # Количество потоков
        'random_seed': 42,                   # Для воспроизводимости
        'file_extension': '.csv'
    }
    
    GROUP_CONFIG = {
        'inc1': {
            'fraction': 1/3,               
            'suffix': 'inc1',
            'description': 'первая треть лог-диапазона'
        },
        'inc2': {
            'fraction': 2/3,             
            'suffix': 'inc2',
            'description': 'первые две трети лог-диапазона'
        },
        'full': {
            'fraction': None,             
            'suffix': 'full',
            'description': 'полные (без обрезания)'
        }
    }
    
    # Создание процессора
    processor = ThreeGroupWellTestProcessor()
    
    # Обработка всех файлов
    results = processor.process_all_files(
        input_dir=CONFIG['input_dir'],
        output_dir=CONFIG['output_dir'],
        n_points=CONFIG['n_points'],
        group_config=GROUP_CONFIG,
        normalize=CONFIG['normalize'],
        overwrite=CONFIG['overwrite'],
        parallel=CONFIG['parallel'],
        n_workers=CONFIG['n_workers'],
        file_extension=CONFIG['file_extension'],
        random_seed=CONFIG['random_seed']
    )
    
    # Сохранение информации о распределении
    processor.save_group_assignment(CONFIG['output_dir'])
    
    # Дополнительная статистика
    print(f"\n{'='*60}")
    print(f"ГОТОВО! Обработанные файлы сохранены в: {CONFIG['output_dir']}")
    print(f"{'='*60}")
    print(f"\nСтруктура выходных файлов:")
    print(f"  *_inc1.csv - первая треть лог-диапазона каждой кривой")
    print(f"  *_inc2.csv - первые две трети лог-диапазона каждой кривой")
    print(f"  *_full.csv - полные кривые (без обрезания)")
    print(f"\nКаждый файл содержит колонки:")
    print(f"  - t_D (время)")
    print(f"  - dP_wD (производная давления)")
    print(f"  - P_wD_gauss (дополнительный столбец, если был в исходных данных)")
    print(f"  - t_D_norm, dP_wD_norm, P_wD_gauss_norm (нормализованные версии)")


if __name__ == "__main__":
    main()
