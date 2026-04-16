from model.reservoir_model import ReservoirModel
import sys


class CommonReservoirModel(ReservoirModel):
    """
    Общая для всех реализация модели пласта

    Можно загрузить датасет из файла
    """
    def F(self, s):
        print('Not supported method!')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        num = sys.argv[1]
        file_name = f'/home/ilnaz/PycharmProjects/datasets/radial_composite/curve/radial_composite_inf_{num}.csv'
    else:
        file_name = '/home/ilnaz/PycharmProjects/datasets/homogeneous_inf_06/curve/homogeneous_fin_1.csv'

    # file_name = '/home/ilnaz/PycharmProjects/datasets/homogeneous_inf_06/curve/homogeneous_fin_1.csv'
    # file_name = '../datasets/homogeneous/curve/homogeneous_fin_20001.csv'

    model = CommonReservoirModel()
    model.load_model(file_name) \
         .visualize('Common model', point_type='*')
