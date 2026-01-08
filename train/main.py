from train.data_preprocess_1d import PressureDataClassificationPreprocessor1D
from train.model import WellTest1DCNN
from train.train import UniversalWellTestTrainer


if __name__ == "__main__":
    # Подготовка данных
    data_preprocess = PressureDataClassificationPreprocessor1D(debug=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()

    X_train = data_preprocess.normalize(X_train)
    X_val = data_preprocess.normalize(X_val)
    X_test = data_preprocess.normalize(X_test)

    class_names = data_preprocess.get_class_names()

    # Инициализация модели
    model = WellTest1DCNN(num_classes=2, input_shape=(128, 2))
    model.compile_model(learning_rate=0.01)

    # Обучение
    trainer = UniversalWellTestTrainer(model.model, class_names)
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        epochs=5,
        initial_lr=0.00001
    )

    # Обработка результатов
    metrics = trainer.evaluate(X_test, y_test)

    sample = X_test[0]
    benchmark = trainer.benchmark_inference(sample, n_iterations=10)

    trainer.plot_training_history(save_path='training_results.png')

    trainer.save_model('my_universal_model', format='both')

    pred_class, probs, confidence = trainer.predict_single(X_test[0])
    print(f"Класс: {pred_class}, Уверенность: {confidence:.2%}")