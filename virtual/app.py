
import streamlit as st
import pickle
import PyPDF2
import docx
import requests
import gdown
from io import BytesIO


import re
# Load clf.pkl from Google Drive
@st.cache_resource
def load_pickle_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    output = BytesIO()
    gdown.download(url, output, quiet=False, fuzzy=True)
    output.seek(0)
    return pickle.load(output)

# Google Drive file ID for clf.pkl
clf_file_id = "1xk8Q3qCTtGGcgyt4DxhXM6rsffnG78Oo"  # ⬅️ Replace with your real ID
tfidf_file_id = "1HPyotft-sYgotNUazpMJb9F9N3gAg-0C"
encoder_file_id = "1EThhP8gyZuPEPhvc2GpNtKjnyxza9XKL"
# Load clf from Drive
@st.cache_resource
def load_clf():
    return load_pickle_from_drive(clf_file_id)

# Load pre-trained model and TF-IDF vectorizer
svc_model = load_clf()
tfidf = load_pickle_from_drive(tfidf_file_id)
le = load_pickle_from_drive(encoder_file_id)




def clean_resume(txt):
    clean_text = re.sub(r'http\S+\s', ' ', txt)  # Remove URLs
    clean_text = re.sub(r'RT|cc', ' ', clean_text)  # Remove retweet markers
    clean_text = re.sub(r'#\S+\s', ' ', clean_text)  # Remove hashtags
    clean_text = re.sub(r'@\S+', ' ', clean_text)  # Remove mentions
    clean_text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@^_`{|}~][\\"""), ' ', clean_text)  # Remove punctuations
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)  # Remove non-ASCII characters
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Replace multiple spaces with a single space
    return clean_text



# extract text from pdf
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

def prediction(input_resume):
    cleaned_text = clean_resume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]

def main():
    st.title("Resume Categorisation App")
    st.title("By Nikhil Sharma(102203474)")
    st.title("and Yatharth Kohli(102203648)")
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Predicted Category")
            category = prediction(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()

