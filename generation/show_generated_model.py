from model.reservoir_model import ReservoirModel


class CommonReservoirModel(ReservoirModel):
    """
    Общая для всех реализация модели пласта

    Можно загрузить датасет из файла
    """
    def F(self, s):
        print('Not supported method!')

if __name__ == '__main__':
    file_name = '../datasets/dataset05/curve/homogeneous_inf_1.csv'

    model = CommonReservoirModel()
    model.load_model(file_name) \
         .visualize('Common model', point_type='*')
