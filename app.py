import streamlit as st
import joblib

# Cargar modelo y vectorizador
model = joblib.load('models/logistic_regression_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Título
st.title("🔍 Detector de Toxicidad")
st.write("Escribe un comentario y el modelo dirá si es tóxico o no.")

# Input
texto = st.text_area("Comentario:", placeholder="Escribe aquí...")

# Botón
if st.button("Analizar"):
    if texto.strip() == "":
        st.warning("Escribe algo primero.")
    else:
        vectorizado = vectorizer.transform([texto])
        prediccion = model.predict(vectorizado)[0]
        
        if prediccion:
            st.error("🔴 TÓXICO")
        else:
            st.success("🟢 NO TÓXICO")