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
        # file_name = f'/home/ilnaz/PycharmProjects/datasets/homogeneous_inf/curve/homogeneous_inf_{num}.csv'
        # file_name = f'/home/ilnaz/PycharmProjects/datasets/homogeneous_inf_04/curve/homogeneous_fin_{num}.csv'
        file_name = f'/home/ilnaz/PycharmProjects/datasets/dual_porosity_inf/curve/dual_porosity_inf_{num}.csv'
    else:
        # file_name = '/home/ilnaz/PycharmProjects/datasets/homogeneous_inf_04/curve/homogeneous_fin_1.csv'
        file_name = '/home/ilnaz/PycharmProjects/datasets/homogeneous_inf_incs_prc/curve/homogeneous_inf_4_inc1.csv'

    file_name = '/home/ilnaz/PycharmProjects/datasets/homogeneous_inf_incs_prc/curve/homogeneous_inf_10_inc1.csv'

    # file_name = '/home/ilnaz/PycharmProjects/datasets/homogeneous_inf/curve/homogeneous_inf_45.csv'
    # file_name = '../datasets/homogeneous/curve/homogeneous_fin_20001.csv'

    model = CommonReservoirModel()
    model.load_model(file_name) \
         .visualize('Common model', point_type='*')

    # model.load_and_preprocess(file_name, t_min=0, t_max=1e4)\
    #      .visualize('Common model', point_type='*')

