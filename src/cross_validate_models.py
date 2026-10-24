from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier


def main():
    print("SmartTicket AI - Cross Validation")

    # Proje klasörünü buluyoruz.
    project_dir = Path(__file__).resolve().parents[1]

    # Yalnızca eğitim verisini okuyoruz.
    train_path = project_dir / "data" / "processed" / "train.csv"
    train = pd.read_csv(train_path)

    X = train["text"].fillna("")
    y = train["type"]

    print("\nEğitim kayıt sayısı:", len(X))

    # 5 parçalı çapraz doğrulama.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Karşılaştıracağımız modeller.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Linear SVM": LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        ),

        "Naive Bayes": MultinomialNB(
            alpha=1.0
        )
    }

    # Baseline performansı.
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline_scores = cross_val_score(
        baseline,
        X,
        y,
        cv=cv,
        scoring="f1_macro"
    )

    print("\nBaseline:")
    print("Ortalama Macro F1:", baseline_scores.mean())
    print("Standart sapma:", baseline_scores.std())

    # Üç modelin çapraz doğrulaması.
    for name, classifier in models.items():

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True
                )
            ),
            ("classifier", classifier)
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="f1_macro"
        )

        print(f"\n--- {name} ---")
        print("Fold skorları:", scores.round(4))
        print("Ortalama Macro F1:", scores.mean())
        print("Standart sapma:", scores.std())


if __name__ == "__main__":
    main()