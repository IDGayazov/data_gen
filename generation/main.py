from typing import Final

from generation.dual_permeability import InfiniteDualPermeabilityModelGenerator, FiniteDualPermeabilityGenerator
from generation.dual_porosity import InfiniteDualPorosityGenerator, FiniteDualPorosityGenerator
from generation.generator import GenerationParams, DataGenerator
from generation.homogeneous import InfiniteHomogeneousGenerator, FiniteHomogeneousGenerator
from generation.radial_composite import InfiniteRadialCompositeGenerator
from generation.utils import clear_folder


def single_generation_homogeneous_inf():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIGMA: Final = 5e-3
    OUTPUT_PATH: Final = '../datasets/new/homogeneous_inf'

    params_train = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 14000, SIGMA, OUTPUT_PATH, 'train')
    params_val = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 2000, SIGMA, OUTPUT_PATH, 'val')
    params_test = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 4000, SIGMA, OUTPUT_PATH, 'test')

    clear_folder(OUTPUT_PATH)

    generators = [
        InfiniteHomogeneousGenerator(params_train),
        InfiniteHomogeneousGenerator(params_val),
        InfiniteHomogeneousGenerator(params_test),
    ]

    for generator in generators:
        generator.generate()

    generator.reset_global_counter()


def single_generation_homogeneous_fin():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIGMA: Final = 5e-3
    OUTPUT_PATH: Final = '../datasets/new/homogeneous_fin'

    params_train = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 14000, SIGMA, OUTPUT_PATH, 'train')
    params_val = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 2000, SIGMA, OUTPUT_PATH, 'val')
    params_test = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 4000, SIGMA, OUTPUT_PATH, 'test')

    clear_folder(OUTPUT_PATH)

    generators = [
        FiniteHomogeneousGenerator(params_train),
        FiniteHomogeneousGenerator(params_val),
        FiniteHomogeneousGenerator(params_test),
    ]

    for generator in generators:
        generator.generate()

    generator.reset_global_counter()


def single_generation_dual_porosity_inf():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIGMA: Final = 5e-3
    OUTPUT_PATH: Final = '../datasets/new/dual_porosity_inf'

    params_train = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 14000, SIGMA, OUTPUT_PATH, 'train')
    params_val = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 2000, SIGMA, OUTPUT_PATH, 'val')
    params_test = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 4000, SIGMA, OUTPUT_PATH, 'test')

    clear_folder(OUTPUT_PATH)

    generators = [
        InfiniteDualPorosityGenerator(params_train),
        InfiniteDualPorosityGenerator(params_val),
        InfiniteDualPorosityGenerator(params_test),
    ]

    for generator in generators:
        generator.generate()

    generator.reset_global_counter()


def single_generation_dual_porosity_fin():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIGMA: Final = 5e-3
    OUTPUT_PATH: Final = '../datasets/new/dual_porosity_fin'

    params_train = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 14000, SIGMA, OUTPUT_PATH, 'train')
    params_val = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 2000, SIGMA, OUTPUT_PATH, 'val')
    params_test = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 4000, SIGMA, OUTPUT_PATH, 'test')

    clear_folder(OUTPUT_PATH)

    generators = [
        FiniteDualPorosityGenerator(params_train),
        FiniteDualPorosityGenerator(params_val),
        FiniteDualPorosityGenerator(params_test)
    ]

    for generator in generators:
        generator.generate()

    generator.reset_global_counter()


def single_generation_dual_permeability_inf():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIGMA: Final = 5e-3
    OUTPUT_PATH: Final = '../datasets/new/dual_permeability_inf'

    params_train = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 14000, SIGMA, OUTPUT_PATH, 'train')
    params_val = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 2000, SIGMA, OUTPUT_PATH, 'val')
    params_test = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 4000, SIGMA, OUTPUT_PATH, 'test')

    clear_folder(OUTPUT_PATH)

    generators = [
        InfiniteDualPermeabilityModelGenerator(params_train),
        InfiniteDualPermeabilityModelGenerator(params_val),
        InfiniteDualPermeabilityModelGenerator(params_test)
    ]

    for generator in generators:
        generator.generate()

    generator.reset_global_counter()


def single_generation_dual_permeability_fin():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIGMA: Final = 5e-3
    OUTPUT_PATH: Final = '../datasets/new/dual_permeability_fin'

    params_train = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 14000, SIGMA, OUTPUT_PATH, 'train')
    params_val = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 2000, SIGMA, OUTPUT_PATH, 'val')
    params_test = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 4000, SIGMA, OUTPUT_PATH, 'test')

    clear_folder(OUTPUT_PATH)

    generators = [
        FiniteDualPermeabilityGenerator(params_train),
        FiniteDualPermeabilityGenerator(params_val),
        FiniteDualPermeabilityGenerator(params_test)
    ]

    for generator in generators:
        generator.generate()

    generator.reset_global_counter()

def single_generation_radial_composite_inf():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIGMA: Final = 5e-3
    OUTPUT_PATH: Final = '../datasets/new/radial_composite_inf'

    params_train = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 14000, SIGMA, OUTPUT_PATH, 'train')
    params_val = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 2000, SIGMA, OUTPUT_PATH, 'val')
    params_test = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 4000, SIGMA, OUTPUT_PATH, 'test')

    clear_folder(OUTPUT_PATH)

    generators = [
        InfiniteRadialCompositeGenerator(params_train),
        InfiniteRadialCompositeGenerator(params_val),
        InfiniteRadialCompositeGenerator(params_test)
    ]

    for generator in generators:
        generator.generate()

    generator.reset_global_counter()

def generation_for_classification():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIGMA: Final = 5e-3
    OUTPUT_PATH: Final = '../datasets/new/class_dataset'

    params_train = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 3500, SIGMA, OUTPUT_PATH, 'train')
    params_val = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 750, SIGMA, OUTPUT_PATH, 'val')
    params_test = GenerationParams(T_MAX_DAYS, POINTS_COUNT, 750, SIGMA, OUTPUT_PATH, 'test')

    clear_folder(OUTPUT_PATH)

    generators = [
        InfiniteHomogeneousGenerator(params_train),
        InfiniteHomogeneousGenerator(params_val),
        InfiniteHomogeneousGenerator(params_test),

        FiniteHomogeneousGenerator(params_train),
        FiniteHomogeneousGenerator(params_val),
        FiniteHomogeneousGenerator(params_test),

        InfiniteDualPorosityGenerator(params_train),
        InfiniteDualPorosityGenerator(params_val),
        InfiniteDualPorosityGenerator(params_test),

        FiniteDualPorosityGenerator(params_train),
        FiniteDualPorosityGenerator(params_val),
        FiniteDualPorosityGenerator(params_test),

        InfiniteDualPermeabilityModelGenerator(params_train),
        InfiniteDualPermeabilityModelGenerator(params_val),
        InfiniteDualPermeabilityModelGenerator(params_test),

        FiniteDualPermeabilityGenerator(params_train),
        FiniteDualPermeabilityGenerator(params_val),
        FiniteDualPermeabilityGenerator(params_test),

        InfiniteRadialCompositeGenerator(params_train),
        InfiniteRadialCompositeGenerator(params_val),
        InfiniteRadialCompositeGenerator(params_test)
    ]

    for generator in generators:
        generator.generate()

    generator.reset_global_counter()

def noize_relation_generation():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIZE: Final = 4000
    BASE_OUTPUT_PATH: Final = '../datasets/dataset'

    params = [
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-7, BASE_OUTPUT_PATH + '07'),
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-6, BASE_OUTPUT_PATH + '06'),
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-5, BASE_OUTPUT_PATH + '05'),
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-4, BASE_OUTPUT_PATH + '04'),
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-3, BASE_OUTPUT_PATH + '03'),
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-2, BASE_OUTPUT_PATH + '02'),
        GenerationParams(T_MAX_DAYS, POINTS_COUNT, SIZE, 5e-1, BASE_OUTPUT_PATH + '01')
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
    # single_generation_homogeneous_inf()
    # single_generation_homogeneous_fin()
    # single_generation_dual_porosity_inf()
    # single_generation_dual_porosity_fin()
    # single_generation_dual_permeability_inf()
    # single_generation_dual_permeability_fin()
    # single_generation_radial_composite_inf()

    generation_for_classification()

    # noize_relation_generation()

if __name__ == "__main__":
    main()
