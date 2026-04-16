from typing import Final

from generation.dual_permeability import InfiniteDualPermeabilityModelGenerator, FiniteDualPermeabilityGenerator
from generation.dual_porosity import InfiniteDualPorosityGenerator, FiniteDualPorosityGenerator
from generation.generator import GenerationParams, DataGenerator
from generation.homogeneous import InfiniteHomogeneousGenerator, FiniteHomogeneousGenerator
from generation.radial_composite import InfiniteRadialCompositeGenerator
from generation.utils import clear_folder


def single_generation():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIZE: Final = 1000
    SIGMA: Final = 5e-4
    OUTPUT_PATH: Final = '../datasets/models'

    params = GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, SIGMA, OUTPUT_PATH)

    clear_folder(OUTPUT_PATH)

    generators = [
        InfiniteHomogeneousGenerator(params),
        FiniteHomogeneousGenerator(params),
        InfiniteDualPorosityGenerator(params),
        FiniteDualPorosityGenerator(params),
        InfiniteDualPermeabilityModelGenerator(params),
        FiniteDualPermeabilityGenerator(params),
        InfiniteRadialCompositeGenerator(params)
    ]

    for generator in generators:
        generator.generate()


def noize_relation_generation():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIZE: Final = 5000
    BASE_OUTPUT_PATH: Final = '../datasets/dataset'

    params = [
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-7, BASE_OUTPUT_PATH + '07'),
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-6, BASE_OUTPUT_PATH + '06'),
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-5, BASE_OUTPUT_PATH + '05'),
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-4, BASE_OUTPUT_PATH + '04'),
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-3, BASE_OUTPUT_PATH + '03')
    ]

    for param in params:
        print('Generating dataset for:', param)

        clear_folder(param.output_path)
        DataGenerator.reset_global_counter()

        generators = [
            InfiniteHomogeneousGenerator(param),
            FiniteHomogeneousGenerator(param),
            InfiniteDualPorosityGenerator(param),
            FiniteDualPorosityGenerator(param),
            InfiniteDualPermeabilityModelGenerator(param),
            FiniteDualPermeabilityGenerator(param),
            InfiniteRadialCompositeGenerator(param)
        ]

        for generator in generators:
            generator.generate()


def main():
    single_generation()
    # noize_relation_generation()

if __name__ == "__main__":
    main()