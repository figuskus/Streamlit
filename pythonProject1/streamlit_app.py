import streamlit as st
import pandas as pd
import time
import matplotlib as plt
import os

# zaczynamy od zaimportowania bibliotek

# st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')
# streamlit jest wykorzystywany do tworzenia aplikacji
# z tego powodu dobrą praktyką jest informowanie użytkownika o postępie, błędach, etc.

# Inne przykłady do wypróbowania:
st.balloons() # animowane balony ;)
# st.error('Błąd!') # wyświetla informację o błędzie
# st.warning('Ostrzeżenie, działa, ale chyba tak sobie...')
# st.info('Informacja...')
# st.success('Udało się!')

st.spinner()
with st.spinner(text='Pracuję...'):
    time.sleep(2)
    st.success('Done')
# możemy dzięki temu "ukryć" późniejsze ładowanie aplikacji

st.title('Lab05. Streamlit :)))')
st.image("logo.jpg")
# title, jak sama nazwa wskazuje, używamy do wyświetlenia tytułu naszej aplikacji

st.header('Translator z angielskiego na niemiecki')
# header to jeden z podtytułów wykorzystywnaych w Streamlit
st.text('Ta aplikacja służy do tłumaczenia tekstu z języka angielskiego na niemiecki, stosując do tego funkcjonalności hugginface')
# text używamy do wyświetlenia dowolnego tekstu. Można korzystać z polskich znaków.


# write używamy również do wyświetlenia tekstu, różnica polega na formatowaniu.

#st.code("st.write()", language='python')
# code może nam się czasami przydać, jeżeli chcielibyśmy pokazać np. klientowi fragment kodu, który wykorzystujemy w aplikacji

# with st.echo():
#     st.write("Echo")
# możemy też to zrobić prościej używając echo - pokazujemy kod i równocześnie go wykonujemy

# df = pd.read_csv("DSP_4.csv", sep=';')
# st.dataframe(df)
# musimy tylko pamiętać o właściwym określeniu separatora (w tym wypadku to średnik)
# masz problem z otworzeniem pliku? sprawdź w jakim katalogu pracujesz i dodaj tam plik (albo co bardziej korzystne - zmień katalog pracy)
# os.getcwd() # pokaż bieżący katalog
# os.chdir("") # zmiana katalogu

st.header('Tłumaczenie z angielskiego na niemiecki')

st.write('Wpisz w pole poniżej tekst do przetłumaczenia i zatwierdź za pomocą przycisku \'Przetłumacz\' ')

with st.spinner("Ładuję..."):
    import streamlit as st
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



# st.subheader('Zadanie do wykonania')
# st.write('Wykorzystaj Huggin Face do stworzenia swojej własnej aplikacji tłumaczącej tekst z języka angielskiego na język niemiecki. Zmodyfikuj powyższy kod dodając do niego kolejną opcję, tj. tłumaczenie tekstu. Informacje potrzebne do zmodyfikowania kodu znajdziesz na stronie Huggin Face - https://huggingface.co/docs/transformers/index')
# st.write('🐞 Dodaj właściwy tytuł do swojej aplikacji, może jakieś grafiki?')
# st.write('🐞 Dodaj krótką instrukcję i napisz do czego służy aplikacja')
# st.write('🐞 Wpłyń na user experience, dodaj informacje o ładowaniu, sukcesie, błędzie, itd.')
# st.write('🐞 Na końcu umieść swój numer indeksu')
# st.write('🐞 Stwórz nowe repozytorium na GitHub, dodaj do niego swoją aplikację, plik z wymaganiami (requirements.txt)')
# st.write('🐞 Udostępnij stworzoną przez siebie aplikację (https://share.streamlit.io) a link prześlij do prowadzącego')
st.write('Autor: s22586')