import shutil
from pathlib import Path

import numpy as np


def extract_scalar(value):
    """
    Извлекает скалярное значение из pandas Series, numpy array или другого контейнера.

    Args:
        value: значение, которое может быть Series, array, list, tuple или скаляром

    Returns:
        скалярное значение
    """
    if hasattr(value, 'iloc'):
        return value.iloc[0]
    elif hasattr(value, 'item'):
        return value.item()
    elif isinstance(value, (list, tuple, np.ndarray)):
        return value[0] if len(value) > 0 else value
    else:
        return value

def clear_folder(folder_path: str):
    path = Path(folder_path)

    if not path.exists():
        print(f'Path: {folder_path} not exists')
        return

    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    print(f'Directory {folder_path} cleared')