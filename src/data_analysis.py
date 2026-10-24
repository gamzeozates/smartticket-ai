from itertools import islice

import pandas as pd
from datasets import load_dataset


def main():
    print("SmartTicket AI - Veri Analizi")
    print("Veri kümesi yükleniyor...")

    dataset = load_dataset(
        "vasu1111/customer-support-tickets",
        split="train",
        streaming=True
    )

    # İlk 500 kaydı alıyoruz.
    records = list(islice(dataset, 500))

    # Pandas DataFrame oluşturuyoruz.
    df = pd.DataFrame(records)

    print("\nVeri boyutu:")
    print(df.shape)

    print("\nSütun isimleri:")
    print(df.columns.tolist())

    print("\nİlk 5 kayıt:")
    print(
        df[
            ["subject", "type", "queue", "priority", "language"]
        ].head()
    )

    print("\nEksik değer sayıları:")
    columns = [
        "subject", "body", "type",
        "queue", "priority", "language"
    ]
    print(df[columns].isnull().sum())

    print("\nTalep türlerinin dağılımı:")
    print(df["type"].value_counts())

    print("\nDil dağılımı:")
    print(df["language"].value_counts())

    # İngilizce kayıtları seçiyoruz.
    df_en = df[df["language"] == "en"].copy()

    print("\nİngilizce kayıt sayısı:")
    print(len(df_en))

    print("\nİngilizce talep türleri:")
    print(df_en["type"].value_counts())

    # Boş metinleri kontrol ediyoruz.
    empty_subjects = (
        df_en["subject"].fillna("").str.strip() == ""
    ).sum()

    empty_bodies = (
        df_en["body"].fillna("").str.strip() == ""
    ).sum()

    print("\nBoş başlık sayısı:", empty_subjects)
    print("Boş açıklama sayısı:", empty_bodies)

    # Tekrarlanan kayıtları kontrol ediyoruz.
    duplicate_count = df_en.duplicated(
        subset=["subject", "body"]
    ).sum()

    print("\nTekrarlanan talep sayısı:", duplicate_count)

    # Başlık ve açıklamayı birleştiriyoruz.
    df_en["text"] = (
        df_en["subject"].fillna("").astype(str)
        + " "
        + df_en["body"].fillna("").astype(str)
    ).str.strip()

    print("\nİlk örnek metin:")
    print(df_en["text"].iloc[0][:300])

    print("\nAnalize hazır veri boyutu:")
    print(df_en[["text", "type"]].shape)


if __name__ == "__main__":
    main() 