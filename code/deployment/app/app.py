import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")


@st.cache_data(ttl=60)
def load_metadata() -> dict:
    response = requests.get(f"{API_URL}/metadata", timeout=5)
    response.raise_for_status()
    return response.json()


def show_car_form(metadata: dict) -> dict | None:
    categories = metadata["categorical"]
    numeric_features = metadata["numeric"]

    with st.form("car"):
        left_column, right_column = st.columns(2)

        with left_column:
            default_model_index = categories["model"].index("Focus")
            car_model = st.selectbox(
                "Ford model",
                categories["model"],
                index=default_model_index,
            )
            year = st.slider(
                "Year",
                min_value=int(numeric_features["year"]["min"]),
                max_value=int(numeric_features["year"]["max"]),
                value=2018,
            )
            transmission = st.selectbox("Transmission", categories["transmission"])
            fuel_type = st.selectbox("Fuel type", categories["fuel_type"])

        with right_column:
            mileage = st.number_input(
                "Mileage (miles)",
                min_value=0,
                value=25000,
                step=1000,
            )
            engine_size = st.number_input(
                "Engine size (litres)",
                min_value=0.1,
                value=1.0,
                step=0.1,
            )
            mpg = st.number_input(
                "Fuel economy (miles per UK gallon)",
                min_value=0.1,
                value=57.7,
                step=0.1,
            )
            tax = st.number_input(
                "Annual road tax (£)",
                min_value=0.0,
                value=145.0,
                step=5.0,
            )

        submitted = st.form_submit_button("Estimate price", type="primary")

    if not submitted:
        return None

    return {
        "model": car_model,
        "year": year,
        "transmission": transmission,
        "fuel_type": fuel_type,
        "mileage": mileage,
        "engine_size": engine_size,
        "mpg": mpg,
        "tax": tax,
    }


def show_prediction(car_details: dict) -> None:
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=car_details,
            timeout=10,
        )
        response.raise_for_status()
        prediction = response.json()
        price = prediction["price_gbp"]

        st.metric("Estimated advertised price", f"£{price:,.0f}")
        st.caption(
            "just a simple estimation"
        )
    except requests.RequestException as error:
        st.error(f"Prediction failed: {error}")


def main() -> None:
    st.set_page_config(page_title="Ford Price Predictor", page_icon="🚗")
    st.title("Ford Used Car Price Predictor")
    st.caption(
        "Estimate an advertised price in GBP from historical UK Ford listings (2020)."
    )

    try:
        metadata = load_metadata()
    except requests.RequestException as error:
        st.error(f"Model API is unavailable: {error}")
        st.stop()

    car_details = show_car_form(metadata)
    if car_details is not None:
        show_prediction(car_details)

    with st.expander("Held-out test metrics"):
        st.json(metadata["metrics"])


if __name__ == "__main__":
    main()
