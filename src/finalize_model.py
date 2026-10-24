from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def main():
    print("SmartTicket AI - Final Model")

    # 1. Proje dizinini belirliyoruz.
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data" / "processed"

    # 2. Hazırlanmış veri kümelerini okuyoruz.
    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")

    # 3. Eğitim ve validation verilerini birleştiriyoruz.
    train_full = pd.concat(
        [train, validation],
        ignore_index=True
    )

    X_train = train_full["text"].fillna("")
    y_train = train_full["type"]

    X_test = test["text"].fillna("")
    y_test = test["type"]

    print("\nEgitim kayit sayisi:", len(X_train))
    print("Test kayit sayisi:", len(X_test))

    # 4. Seçtiğimiz modeli oluşturuyoruz.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # 5. Nihai modelimizi eğitiyoruz.
    print("\nFinal model egitiliyor...")
    model.fit(X_train, y_train)

    # 6. Ayrı tutulan test verisiyle değerlendiriyoruz.
    predictions = model.predict(X_test)

    print("\n--- FINAL TEST SONUCLARI ---")

    print("Accuracy:", accuracy_score(y_test, predictions))

    print(
        "Macro F1:",
        f1_score(y_test, predictions, average="macro")
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    labels = sorted(y_train.unique())

    print("\nConfusion Matrix:")
    print("Sinif sirasi:", labels)
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=labels
        )
    )

    # 7. Eğitilmiş modeli dosyaya kaydediyoruz.
    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "ticket_classifier.joblib"

    joblib.dump(model, model_path)

    print("\nModel kaydedildi:")
    print(model_path)


if __name__ == "__main__":
    main()