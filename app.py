import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor


st.title("Video Game Sales Prediction using AdaBoost Regressor")


df = pd.read_csv("vgsales.csv")


df = df.dropna()


X = df[['Platform', 'Year', 'Genre', 'Publisher']]
y = df['Global_Sales']


le_platform = LabelEncoder()
le_genre = LabelEncoder()
le_publisher = LabelEncoder()

X['Platform'] = le_platform.fit_transform(X['Platform'])
X['Genre'] = le_genre.fit_transform(X['Genre'])
X['Publisher'] = le_publisher.fit_transform(X['Publisher'])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = AdaBoostRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


platform = st.selectbox(
    "Platform",
    le_platform.classes_
)

year = st.number_input(
    "Year",
    min_value=1980,
    max_value=2025,
    value=2010
)

genre = st.selectbox(
    "Genre",
    le_genre.classes_
)

publisher = st.selectbox(
    "Publisher",
    le_publisher.classes_
)


platform_encoded = le_platform.transform([platform])[0]
genre_encoded = le_genre.transform([genre])[0]
publisher_encoded = le_publisher.transform([publisher])[0]


if st.button("Predict Global Sales"):

    input_data = pd.DataFrame({
        'Platform': [platform_encoded],
        'Year': [year],
        'Genre': [genre_encoded],
        'Publisher': [publisher_encoded]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Global Sales : {prediction[0]:.2f} Million"
    )