from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator


# Model dosyamızın konumu
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "ticket_classifier.joblib"


# API'ye gönderilecek veri
class TicketRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()

        if len(cleaned) < 3:
            raise ValueError("Message is too short.")

        return cleaned


# API'den dönecek veri
class TicketResponse(BaseModel):
    prediction: str
    scores: dict[str, float]
    note: str


# Uygulama başlarken modeli yüklüyoruz.
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    app.state.model = joblib.load(MODEL_PATH)

    print("SmartTicket AI model loaded.")

    yield

    del app.state.model


# API uygulamamızı oluşturuyoruz.
app = FastAPI(
    title="SmartTicket AI API",
    description="AI-powered support ticket classification",
    version="1.0.0",
    lifespan=lifespan,
)


# Ana sayfa
@app.get("/")
def home():
    return {
        "message": "SmartTicket AI API is running",
        "documentation": "/docs",
    }


# Sistem sağlık kontrolü
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
    }


# Yapay zeka tahmin servisi
@app.post("/predict", response_model=TicketResponse)
def predict_ticket(ticket: TicketRequest):

    model = app.state.model

    # Kategori tahmini
    prediction = str(model.predict([ticket.text])[0])

    # Kategori skorları
    probabilities = model.predict_proba(
        [ticket.text]
    )[0]

    scores = {
        str(category): round(float(score), 4)
        for category, score in zip(
            model.classes_,
            probabilities
        )
    }

    return TicketResponse(
        prediction=prediction,
        scores=scores,
        note=(
            "English-only prototype. "
            "Scores are not calibrated confidence values."
        ),
    )