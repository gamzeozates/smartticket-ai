from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def main():
    print("SmartTicket AI - Model Karsilastirma")

    # 1. Proje klasörünü belirliyoruz.
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data" / "processed"

    # 2. Eğitim ve doğrulama verilerini okuyoruz.
    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")

    X_train = train["text"].fillna("")
    y_train = train["type"]

    X_val = validation["text"].fillna("")
    y_val = validation["type"]

    print("\nTraining:", len(X_train))
    print("Validation:", len(X_val))

    # 3. Karşılaştırılacak algoritmalar.
    algorithms = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
        "Naive Bayes": MultinomialNB(
            alpha=1.0,
        ),
    }

    results = []
    best_name = None
    best_model = None
    best_score = -1

    # 4. Baseline modelimizi hesaplıyoruz.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    baseline_pred = baseline.predict(X_val)

    results.append({
        "Model": "Baseline",
        "Accuracy": accuracy_score(y_val, baseline_pred),
        "Macro F1": f1_score(
            y_val, baseline_pred, average="macro"
        ),
    })

    # 5. Üç farklı modeli eğitiyoruz.
    for name, classifier in algorithms.items():

        print(f"\nModel egitiliyor: {name}")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
        })

        if macro_f1 > best_score:
            best_score = macro_f1
            best_name = name
            best_model = pipeline

    # 6. Sonuçları karşılaştırıyoruz.
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Macro F1",
        ascending=False,
    )

    print("\n--- MODEL KARSILASTIRMA SONUCLARI ---")
    print(results_df.to_string(index=False))

    # 7. En başarılı modelin ayrıntılı sonuçları.
    print("\nEn iyi model:", best_name)

    best_predictions = best_model.predict(X_val)

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        best_predictions,
        zero_division=0,
    ))

    print("\nConfusion Matrix:")
    labels = sorted(y_train.unique())
    print("Sinif sirasi:", labels)
    print(confusion_matrix(
        y_val,
        best_predictions,
        labels=labels,
    ))


if __name__ == "__main__":
    main()