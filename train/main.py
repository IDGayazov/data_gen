from matplotlib import pyplot as plt

from train.data_preprocess_1d import PressureDataClassificationPreprocessor1D
from train.cnn_model import WellTest1DCNN
from train.lstm_model import WellTestLSTM
from train.train import WellTestTrainer


def single_train():
    # Подготовка данных
    data_preprocess = PressureDataClassificationPreprocessor1D(data_dir='../datasets/dataset05/curve', debug=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()
    class_names = data_preprocess.get_class_names()
    num_classes = len(class_names)

    X_train = data_preprocess.normalize(X_train)
    X_val = data_preprocess.normalize(X_val)
    X_test = data_preprocess.normalize(X_test)

    # Инициализация модели
    model = WellTest1DCNN(num_classes=num_classes, input_shape=(128, 2))

    # Обучение
    trainer = WellTestTrainer(model.model, class_names)
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        epochs=20,
        initial_lr=0.00001
    )

    # Обработка результатов
    metrics = trainer.evaluate(X_test, y_test)

    sample = X_test[0]
    benchmark = trainer.benchmark_inference(sample, n_iterations=10)

    trainer.plot_training_history(save_path='training_results.png')

    cm = trainer.plot_confusion_matrix(X_test, y_test,
                                   normalize=True,
                                   save_path='confusion_matrix.png')

    trainer.save_model('my_universal_model', format='both')

    pred_class, probs, confidence = trainer.predict_single(X_test[0])
    print(f"Класс: {pred_class}, Уверенность: {confidence:.2%}")
    print(f"Истинный класс: {y_test[0]}")

def train_with_noize_variance():
    dataset_dirs = [
        './datasets/dataset03/curve',
        './datasets/dataset04/curve',
        './datasets/dataset05/curve',
        './datasets/dataset06/curve',
        './datasets/dataset07/curve'
    ]

    loss = []
    accuracy = []
    noize = [5e-3, 5e-4, 5e-5, 5e-6, 5e-7]

    for dir in dataset_dirs:
        noize_range = dir.split('/')[2][-2:]

        data_preprocess = PressureDataClassificationPreprocessor1D(data_dir=dir)
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()
        class_names = data_preprocess.get_class_names()
        num_classes = len(class_names)

        X_train = data_preprocess.normalize(X_train)
        X_val = data_preprocess.normalize(X_val)
        X_test = data_preprocess.normalize(X_test)

        model = WellTest1DCNN(num_classes=num_classes, input_shape=(128, 2))

        trainer = WellTestTrainer(model.model, class_names)
        history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=32,
            epochs=20,
            initial_lr=0.00001
        )

        metrics = trainer.evaluate(X_test, y_test)
        trainer.plot_training_history(save_path=f'training_results_{noize_range}.png')
        cm = trainer.plot_confusion_matrix(X_test, y_test,
                                       normalize=True,
                                       save_path=f'confusion_matrix_{noize_range}.png')
        loss.append(metrics['loss'])
        accuracy.append(metrics['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.semilogx(noize, loss, 'r-o', linewidth=2, markersize=8)
    ax1.set_title('Зависимость ошибки от уровня шума', fontsize=14)
    ax1.set_xlabel('Уровень шума', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(noize, accuracy, 'g-s', linewidth=2, markersize=8)
    ax2.set_title('Зависимость точности от уровня шума', fontsize=14)
    ax2.set_xlabel('Уровень шума', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def lstm_training():
    # Подготовка данных
    data_preprocess = PressureDataClassificationPreprocessor1D(data_dir='../datasets/dataset05/curve', debug=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()
    class_names = data_preprocess.get_class_names()
    num_classes = len(class_names)

    X_train = data_preprocess.normalize(X_train)
    X_val = data_preprocess.normalize(X_val)
    X_test = data_preprocess.normalize(X_test)

    # Инициализация модели
    model = WellTestLSTM(num_classes=num_classes, input_shape=(128, 2))

    # Обучение
    trainer = WellTestTrainer(model.model, class_names)
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        epochs=10,
        initial_lr=0.00001
    )

    # Обработка результатов
    metrics = trainer.evaluate(X_test, y_test)

    sample = X_test[0]
    benchmark = trainer.benchmark_inference(sample, n_iterations=10)

    trainer.plot_training_history(save_path='training_results.png')

    cm = trainer.plot_confusion_matrix(X_test, y_test,
                                       normalize=True,
                                       save_path='confusion_matrix.png')

    trainer.save_model('my_universal_model', format='both')

    pred_class, probs, confidence = trainer.predict_single(X_test[0])
    print(f"Класс: {pred_class}, Уверенность: {confidence:.2%}")
    print(f"Истинный класс: {y_test[0]}")


if __name__ == "__main__":
    single_train()
    # train_with_noize_variance()
    # lstm_training()