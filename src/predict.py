from pathlib import Path
import joblib


# Projemizin ana klasörünü buluyoruz.
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Eğitilmiş modelin dosya yolu.
MODEL_PATH = PROJECT_DIR / "models" / "ticket_classifier.joblib"


def load_model():
    """Kaydedilmiş makine öğrenmesi modelini yükler."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model dosyasi bulunamadi: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


def classify_message(model, message: str):
    """Yeni bir destek mesajını sınıflandırır."""

    prediction = model.predict([message])[0]

    # Her kategoriye ait model skorlarını alıyoruz.
    probabilities = model.predict_proba([message])[0]

    scores = sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    return prediction, scores


def main():
    print("SmartTicket AI - Tahmin Sistemi")
    print("Model yukleniyor...")

    model = load_model()

    print("Model basariyla yuklendi!")
    print("Ingilizce bir destek talebi yazabilirsiniz.")
    print("Cikmak icin 'exit' yazin.")

    while True:
        message = input("\nDestek talebi: ").strip()

        if message.lower() == "exit":
            print("Program kapatiliyor...")
            break

        if not message:
            print("Lutfen bir mesaj girin.")
            continue

        prediction, scores = classify_message(
            model, message
        )

        print("\nTahmin edilen kategori:", prediction)

        print("\nKategori skorlari:")

        for category, score in scores:
            print(f"{category}: %{score * 100:.2f}")


if __name__ == "__main__":
    main()