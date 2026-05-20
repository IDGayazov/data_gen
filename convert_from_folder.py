import tensorflow as tf
import tf2onnx
import onnx
from pathlib import Path

tf.config.set_visible_devices([], 'GPU')

# ── Настройки ──────────────────────────────────────────────────────────────
INPUT_DIR  = Path("/home/ilnaz/PycharmProjects/convert_model/models")  # папка с моделями
OUTPUT_DIR = Path("/home/ilnaz/PycharmProjects/convert_model/onnx")    # куда сохранять .onnx
INPUT_SHAPE = [None, 128, 2]   # измените под вашу архитектуру
INPUT_DTYPE = tf.float32
OPSET       = 15
# ───────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ищем все поддерживаемые форматы Keras
PATTERNS = ["*.keras", "*.h5", "*.hdf5"]
model_paths = [p for pattern in PATTERNS for p in INPUT_DIR.glob(pattern)]

# SavedModel-директории (папка содержит saved_model.pb)
model_paths += [p for p in INPUT_DIR.iterdir()
                if p.is_dir() and (p / "saved_model.pb").exists()]

if not model_paths:
    print(f"⚠️  Модели не найдены в {INPUT_DIR}")
else:
    print(f"🔍 Найдено моделей: {len(model_paths)}\n")

ok, fail = 0, 0

for model_path in sorted(model_paths):
    print(f"▶ Конвертирую: {model_path.name}")
    try:
        # Загрузка
        model = tf.keras.models.load_model(model_path)

        # Сигнатура входа
        input_signature = [
            tf.TensorSpec(shape=INPUT_SHAPE, dtype=INPUT_DTYPE, name="input_layer_1")
        ]

        # Конвертация
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=OPSET
        )

        # Сохранение — имя файла совпадает с исходным, расширение → .onnx
        out_path = OUTPUT_DIR / (model_path.stem + ".onnx")
        onnx.save(onnx_model, str(out_path))

        print(f"  ✅ Сохранено → {out_path}\n")
        ok += 1

    except Exception as e:
        print(f"  ❌ Ошибка: {e}\n")
        fail += 1

print(f"═══════════════════════════════")
print(f"Готово: {ok} успешно, {fail} ошибок")
