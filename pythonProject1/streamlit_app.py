import streamlit as st
import time
import os


st.balloons() # animowane balony ;)

st.spinner()
with st.spinner(text='Pracuję...'):
    time.sleep(2)
    st.success('Done')

st.title('Lab05. Streamlit :)))')
st.image("logo.jpg")

st.header('Translator z angielskiego na niemiecki')
st.text('Ta aplikacja służy do tłumaczenia tekstu z języka angielskiego na niemiecki, stosując do tego funkcjonalności hugginface')

st.header('Tłumaczenie z angielskiego na niemiecki')

st.write('Wpisz w pole poniżej tekst do przetłumaczenia i zatwierdź za pomocą przycisku \'Przetłumacz\' ')

with st.spinner("Ładuję..."):
    from transformers import MarianMTModel, MarianTokenizer

def translate_text(text_to_translate):
    model_name = "Helsinki-NLP/opus-mt-en-de"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    input_ids = tokenizer.encode(text_to_translate, return_tensors="pt")
    translation = model.generate(input_ids, max_length=50, num_return_sequences=1)
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return translated_text



input_text = st.text_area("Wpisz tekst po angielsku:")

if st.button("Przetłumacz"):
    with st.spinner("Tłumacze..."):
        translated_text = translate_text(input_text)
        st.write("Tłumaczenie:", translated_text)

st.write('Autor: s22586')
