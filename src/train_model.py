from itertools import islice

import pandas as pd
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

def main():
    print("SmartTicket AI - Model Eğitimi")
    print("Veri kümesi hazırlanıyor...")

    # 1. İlk 500 kaydı okuyoruz.
    dataset = load_dataset(
        "vasu1111/customer-support-tickets",
        split="train",
        streaming=True,
    )

    records = list(islice(dataset, 500))
    df = pd.DataFrame(records)

    # 2. İngilizce kayıtları seçiyoruz.
    df = df[df["language"] == "en"].copy()

    # 3. Boş kayıtları temizliyoruz.
    df = df.dropna(
        subset=["subject", "body", "type"]
    )

    # 4. Tekrarlanan talepleri çıkarıyoruz.
    df = df.drop_duplicates(
        subset=["subject", "body"]
    )

    # 5. Metin sütununu hazırlıyoruz.
    df["text"] = (
        df["subject"].astype(str)
        + " "
        + df["body"].astype(str)
    ).str.strip()

    df = df[df["text"] != ""]

    # 6. Girdi ve hedef değişkenlerini ayırıyoruz.
    X = df["text"]
    y = df["type"]

    print("\nToplam kayıt:", len(df))
    print("\nSınıf dağılımı:")
    print(y.value_counts())

    # 7. Eğitim ve test kümeleri.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nEğitim kayıt sayısı:", len(X_train))
    print("Test kayıt sayısı:", len(X_test))

    # 8. Model işlem hattını oluşturuyoruz.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    # 9. Model eğitimi.
    print("\nModel eğitiliyor...")
    model.fit(X_train, y_train)

    # 10. Test verileri için tahmin.
    predictions = model.predict(X_test)

    # 11. Başarı ölçümü.
    print("\nModel Accuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    labels = sorted(y.unique())
    print("Sınıf sırası:", labels)
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        )
    )
        # 12. Baseline model oluşturuyoruz.
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    # 13. İki modeli karşılaştırıyoruz.
    print("\n--- MODEL KARŞILAŞTIRMASI ---")

    print("\nBaseline Accuracy:")
    print(accuracy_score(y_test, baseline_predictions))

    print("\nBaseline Macro F1:")
    print(
        f1_score(
            y_test,
            baseline_predictions,
            average="macro"
        )
    )

    print("\nSmartTicket AI Macro F1:")
    print(
        f1_score(
            y_test,
            predictions,
            average="macro"
        )
    )
        # 14. Yanlış tahminleri inceliyoruz.
    results = pd.DataFrame({
        "text": X_test.to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predictions
    })

    errors = results[
        results["actual"] != results["predicted"]
    ]

    print("\n--- HATALI TAHMİNLER ---")
    print("Toplam hata:", len(errors))

    print("\nKarıştırılan kategoriler:")
    print(
        errors.groupby(
            ["actual", "predicted"]
        ).size()
    )

    print("\nİlk 5 hatalı örnek:")

    for _, row in errors.head(5).iterrows():
        print("\nGerçek:", row["actual"])
        print("Tahmin:", row["predicted"])
        print("Metin:", row["text"][:200])


if __name__ == "__main__":
    main()