from pathlib import Path
from itertools import islice

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def main():
    print("SmartTicket AI - Veri Hazırlama")

    # 1. Daha fazla veri okuyoruz.
    dataset = load_dataset(
        "vasu1111/customer-support-tickets",
        split="train",
        streaming=True
    )

    records = list(islice(dataset, 5000))
    df = pd.DataFrame(records)

    print("\nİlk okunan kayıt:", len(df))

    # 2. İngilizce verileri seçiyoruz.
    df = df[df["language"] == "en"].copy()

    # 3. Eksik kayıtları temizliyoruz.
    df = df.dropna(
        subset=["subject", "body", "type"]
    )

    # 4. Metin sütununu hazırlıyoruz.
    df["text"] = (
        df["subject"].astype(str)
        + " "
        + df["body"].astype(str)
    ).str.strip()

    df = df[df["text"] != ""]

    # 5. Tekrarlanan metinleri kaldırıyoruz.
    df = df.drop_duplicates(subset=["text"])

    # Sadece gerekli sütunları tutuyoruz.
    df = df[["text", "type"]].copy()

    print("\nTemiz veri sayısı:", len(df))
    print("\nSınıf dağılımı:")
    print(df["type"].value_counts())

    # 6. Test verisini ayırıyoruz (%15).
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["type"]
    )

    # 7. Kalan veriden validation ayırıyoruz.
    train, validation = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=42,
        stratify=train_val["type"]
    )

    # 8. Verileri kaydediyoruz.
    output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(
        output_dir / "validation.csv", index=False
    )
    test.to_csv(output_dir / "test.csv", index=False)

    print("\n--- VERİ AYIRMA SONUÇLARI ---")
    print("Training:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))

    print("\nDosyalar başarıyla oluşturuldu.")


if __name__ == "__main__":
    main()