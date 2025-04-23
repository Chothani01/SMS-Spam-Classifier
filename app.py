import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i) 
    text = y[:] 
    y.clear() 
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y) # to give string as output

def main():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))

    st.title("Email/SMS Spam Classifier")
    text = st.text_area("Enter the message")
    
    if st.button('Predict'):

        transformed_sms = transform_text(text)

        input_data = tfidf.transform([transformed_sms])

        output = model.predict(input_data)
        if output==0:
            st.success("Not Spam message")
        else:
            st.error("Spam message")
    
if __name__=='__main__':
    main()