from src.classifier import classify_ticket


def main():
    print("SmartTicket AI sistemine hoş geldiniz!")

    message = input("Destek talebinizi yazın: ")

    category = classify_ticket(message)

    print(f"Tahmin edilen kategori: {category}")


if __name__ == "__main__":
    main()