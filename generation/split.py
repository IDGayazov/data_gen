import numpy as np

def make_param_train_val_lists(start, end, n_steps, split_type='log'):
    if split_type == 'lin':
        points = np.linspace(start, end, n_steps + 1)
    else:
        points = np.logspace(np.log10(start), np.log10(end), n_steps + 1)

    list1 = []
    list2 = []

    for i in range(n_steps):
        interval = (points[i], points[i + 1])
        if i % 2 == 0:
            list1.append(interval)
        else:
            list2.append(interval)

    return list1, list2


if __name__ == "__main__":
    list1, list2 = make_param_train_val_lists(1e-8, 1e-5, 16)

    print("Список 1 (чётные интервалы):")
    for item in list1:
        print(f"({item[0]:.2e}, {item[1]:.2e})")

    print("\nСписок 2 (нечётные интервалы):")
    for item in list2:
        print(f"({item[0]:.2e}, {item[1]:.2e})")
