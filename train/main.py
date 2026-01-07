from train.data_preprocess_1d import PressureDataClassificationPreprocessor1D
from train.model import WellTest1DCNN
from train.train import UniversalWellTestTrainer

if __name__ == "__main__":
    model = WellTest1DCNN(num_classes=2, input_shape=(2, 128))
    model.compile_model(learning_rate=0.01)

    trainer = UniversalWellTestTrainer(model.model)

    data_preprocess = PressureDataClassificationPreprocessor1D(debug=False)
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess.get_dataset()

    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        epochs=300,
        initial_lr=0.01
    )

    metrics = trainer.evaluate(X_test, y_test)

    sample = X_test[0]
    benchmark = trainer.benchmark_inference(sample, n_iterations=10)

    trainer.plot_training_history(save_path='training_results.png')

    trainer.save_model('my_universal_model', format='both')

    pred_class, probs, confidence = trainer.predict_single(X_test[0])
    print(f"Класс: {pred_class}, Уверенность: {confidence:.2%}")