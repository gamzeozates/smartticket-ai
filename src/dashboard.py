import os
import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="SmartTicket AI",
    page_icon="🎫",
    layout="wide",
)

st.title("SmartTicket AI")
st.subheader("Intelligent Customer Support Classification")

st.write(
    "Enter a customer support message and let "
    "our machine learning model classify it."
)

st.info(
    "This prototype currently supports English messages only."
)


# API'nin çalışıp çalışmadığını kontrol ediyoruz.
def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )
        response.raise_for_status()

        return response.json().get("model_loaded") is True

    except (requests.exceptions.RequestException, ValueError):
        return False


api_ready = check_api()

with st.sidebar:
    st.header("System Information")

    if api_ready:
        st.success("API Connected")
    else:
        st.error("API Disconnected")

    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Text Processing:** TF-IDF")
    st.markdown("**Language:** English")

    st.divider()

    st.markdown("**Supported Categories**")
    st.write("Incident")
    st.write("Problem")
    st.write("Request")
    st.write("Change")


# Kullanıcı giriş alanı
st.header("Analyze Support Ticket")

with st.form("ticket_form"):
    message = st.text_area(
        "Customer Support Message",
        placeholder="Enter your support ticket here...",
        height=160,
        max_chars=5000,
    )

    submitted = st.form_submit_button(
        "Classify Ticket",
        type="primary",
    )


# Tahmin işlemi
if submitted:

    if len(message.strip()) < 3:
        st.warning("Please enter at least 3 characters.")

    elif not api_ready:
        st.error(
            "FastAPI is not running. "
            "Please start the backend server."
        )

    else:
        try:
            with st.spinner("Analyzing ticket..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": message.strip()},
                    timeout=15,
                )

                response.raise_for_status()
                result = response.json()

        except requests.exceptions.RequestException as error:
            st.error(f"API request failed: {error}")

        except ValueError:
            st.error("The API returned invalid JSON.")

        else:
            prediction = result["prediction"]
            scores = result["scores"]

            st.divider()
            st.header("Classification Results")

            st.success(
                f"Predicted Category: {prediction}"
            )

            # Skorları grafik için hazırlıyoruz.
            chart_data = pd.DataFrame(
                [
                    {
                        "Category": category,
                        "Score (%)": score * 100,
                    }
                    for category, score in scores.items()
                ]
            )

            chart_data = chart_data.sort_values(
                by="Score (%)",
                ascending=False,
            )

            st.subheader("Category Scores")

            st.bar_chart(
                chart_data,
                x="Category",
                y="Score (%)",
                horizontal=True,
            )

            st.caption(
                "These are model scores, not calibrated "
                "confidence values."
            )

            with st.expander("Detailed Scores"):
                st.dataframe(
                    chart_data,
                    hide_index=True,
                )

            st.caption(result.get("note", ""))