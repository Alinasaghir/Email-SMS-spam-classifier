import streamlit as st
import base64
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix

nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
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

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Function to encode the image to base64
def get_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

# Encode the local image
background_image = get_base64("background.jpg")

# Apply custom CSS
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Raleway:ital,wght@0,100..900;1,100..900&display=swap');

    .stApp {{
        background-image: url("data:image/jpg;base64,{background_image}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    .stApp > div:first-child {{
        padding-top: 0;
        margin-top: 0;
    }}

    [data-testid = "stHeader"]{{
        background-color: rgba(0,0,0,0);
        color: rgb(255, 255, 255);
    }}

    [data-testid = "stHeading"] {{
        background: rgba(0, 0, 0, 0.5);
        font-size: 2em;
        text-align: center;
        padding: 10px;
        margin-top: -50px; 
        border-radius: 10px;
        height: 105px;
    }}

    h1#email-sms-spam-classifier {{
        font-family: "Raleway";
        font-weight: 300;   
        font-size: 1em;
        color: white; 
        text-align: center;
        text-shadow: 
            0 0 5px rgba(255, 255, 255, 0.8), 
            0 0 10px rgba(255, 255, 255, 0.6), 
            0 0 15px rgba(0, 0, 255, 0.5), 
            0 0 20px rgba(255, 0, 255, 0.4), 
            0 0 25px rgba(0, 255, 255, 0.3),
            0 0 30px rgba(255, 0, 255, 0.2);
    }}

    [data-testid = "stTextArea"] {{
        margin-top: 30px;
        background-color: #f0f0f0;
        border-radius: 10px;
        border: 2px solid #cccccc;
        padding: 10px;
        font-size: 1.2em;
    }}

    [data-testid = "baseButton-secondary"] {{
        background: linear-gradient(45deg, #ff7eb9, #ffd700, #ffa500); /* Gradient background */
        border-radius: 100px;
        box-shadow: rgba(255, 126, 185, .2) 0 -25px 18px -14px inset,rgba(255, 215, 0, .15) 0 1px 2px,rgba(255, 165, 0, .15) 0 2px 4px,rgba(255, 126, 0, .15) 0 4px 8px,rgba(255, 126, 0, .15) 0 8px 16px,rgba(255, 126, 0, .15) 0 16px 32px;
        color: black; /* Text color */
        cursor: pointer;
        display: inline-block;
        font-family: CerebriSans-Regular,-apple-system,system-ui,Roboto,sans-serif;
        padding: 12px 30px; /* Adjust button size here */
        text-align: center;
        text-decoration: none;
        transition: all 250ms;
        border: 0;
        font-size: 18px; /* Adjust button font size */
        user-select: none;
        -webkit-user-select: none;
        touch-action: manipulation;

    }}

    [data-testid = "stMarkdownContainer"] p {{
        font-weight: 900;
    }}

    [data-testid = "stButton"] {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }}

    [data-testid = "baseButton-secondary"]:hover {{
        box-shadow: rgba(255, 126, 185, .35) 0 -25px 18px -14px inset,rgba(255, 215, 0, .25) 0 1px 2px,rgba(255, 165, 0, .25) 0 2px 4px,rgba(255, 126, 0, .25) 0 4px 8px,rgba(255, 126, 0, .25) 0 8px 16px,rgba(255, 126, 0, .25) 0 16px 32px;
        color: black;
    }}

    [data-testid = "baseButton-secondary"] p:active {{
        color: black;
    }}

    [data-testid = "baseButton-secondary"] p:visited {{
        color: black;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Email / SMS Spam Classifier")

input_sms = st.text_area("", placeholder="Enter the message", label_visibility="collapsed")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown(
            '<h3 style="background-color: rgba(255, 0, 0, 0.6); color: white; text-align: center; padding: 10px; border-radius: 10px; width: 50%; margin: 30px auto;">Spam</h3>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<h3 style="background-color: rgba(0, 128, 0, 0.6); color: white; text-align: center; padding: 10px; border-radius: 10px; width: 50%; margin: 30px auto;">Not Spam</h3>',
            unsafe_allow_html=True
        )

