def classify_ticket(message: str) -> str:
    """Destek talebini anahtar kelimelere göre sınıflandırır."""

    message = message.lower()

    if "şifre" in message or "hesap" in message:
        return "Hesap İşlemleri"

    elif "ödeme" in message or "sipariş" in message:
        return "Ödeme / Sipariş"

    elif "hata" in message or "kapanıyor" in message:
        return "Teknik Destek"

    else:
        return "Diğer"