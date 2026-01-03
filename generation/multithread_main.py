from generation.dual_permeability import FiniteDualPermeabilityGenerator, InfiniteDualPermeabilityModelGenerator
from generation.dual_porosity import InfiniteDualPorosityGenerator, FiniteDualPorosityGenerator
from generation.homogeneous import InfiniteHomogeneousGenerator, FiniteHomogeneousGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed

from generation.radial_composite import InfiniteRadialCompositeGenerator


def run_generator(generator):
    """Функция для запуска генератора"""
    print(f"Запускается {generator.__class__.__name__}")
    generator.generate()
    return generator.__class__.__name__


def main():
    generators = [
        InfiniteHomogeneousGenerator(30, 128, 1),
        FiniteHomogeneousGenerator(30, 128, 1),
        InfiniteDualPorosityGenerator(30, 128, 1),
        FiniteDualPorosityGenerator(30, 128, 1),
        InfiniteDualPermeabilityModelGenerator(30, 128, 1),
        FiniteDualPermeabilityGenerator(30, 128, 1),
        InfiniteRadialCompositeGenerator(30, 128, 1)
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_generator, gen): gen.__class__.__name__
            for gen in generators
        }

        for future in as_completed(futures):
            generator_name = futures[future]
            try:
                result = future.result()
                print(f"Генератор {result} успешно завершил работу")
            except Exception as e:
                print(f"Ошибка в генераторе {generator_name}: {e}")


if __name__ == "__main__":
    main()