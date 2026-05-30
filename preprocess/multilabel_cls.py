import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

class MultilabelEncoder:
    def __init__(self):
        # Определяем все возможные метки
        self.all_labels = [
            'homogeneous',
            'dual_porosity', 
            'dual_permeability',
            'radial_composite',
            'inf',
            'fin',
            'inc1',
            'inc2',
            'full'
        ]
        
        # Создаем маппинг label -> индекс
        self.label_to_idx = {label: i for i, label in enumerate(self.all_labels)}
        
        self.n_classes = len(self.all_labels)
        
    def encode(self, reservoir, boundary, suffix):
        """
        Преобразование трех меток в multilabel вектор
        
        Parameters:
        -----------
        reservoir : str (например, 'homogeneous')
        boundary : str (например, 'infinite')  
        suffix : str (например, 'inc1')
        
        Returns:
        --------
        np.array : бинарный вектор длины 8
        """
        # Инициализируем нулевой вектор
        y = np.zeros(self.n_classes, dtype=np.int8)
        
        # Устанавливаем 1 для соответствующих меток
        y[self.label_to_idx[reservoir]] = 1
        y[self.label_to_idx[boundary]] = 1
        y[self.label_to_idx[suffix]] = 1
        
        return y
    
    def encode_batch(self, reservoirs, boundaries, suffixes):
        """
        Преобразование батча меток
        """
        y_batch = []
        for r, b, s in zip(reservoirs, boundaries, suffixes):
            y_batch.append(self.encode(r, b, s))
        return np.array(y_batch)
    
    def decode(self, y_vector, threshold=0.5):
        """
        Обратное преобразование: из бинарного вектора в метки
        """
        if isinstance(y_vector, np.ndarray):
            y_binary = (y_vector > threshold).astype(int)
        else:
            y_binary = y_vector
            
        # Получаем индексы, где значение = 1
        indices = np.where(y_binary == 1)[0]
        
        # Преобразуем индексы в названия меток
        labels = [self.all_labels[i] for i in indices]
        
        # Разделяем на три категории
        reservoir_labels = [l for l in labels if l in ['homogeneous', 'dual_porosity', 'dual_permeability', 'radial_composite']]
        boundary_labels = [l for l in labels if l in ['inf', 'fin']]
        suffix_labels = [l for l in labels if l in ['inc1', 'inc2', 'full']]
        
        return {
            'reservoir': reservoir_labels[0] if reservoir_labels else None,
            'boundary': boundary_labels[0] if boundary_labels else None,
            'suffix': suffix_labels[0] if suffix_labels else None
        }

if __name__ == "__main__":
    encoder = MultilabelEncoder()

    y = encoder.encode('homogeneous', 'inf', 'full')
    print(f"Вектор меток: {y}")
    print(f"Длина вектора: {len(y)}")
    print(f"Сумма (кол-во активных меток): {y.sum()}")

    decoded = encoder.decode(y)
    print(f"Декодировано: {decoded}")