import os

# Replace "YOUR_API_KEY" with your actual API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyACHba0Frja_dsWH1c6AMx1yUxYhA03LH0"

import streamlit as st
from PyPDF2 import PdfReader #library to read pdf files
from langchain.text_splitter import RecursiveCharacterTextSplitter#library to split pdf files
import os
import genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings #to embed the text
import google.generativeai as genai


 #to chain the prompts
from langchain.prompts import PromptTemplate #to create prompt templates
#from dotenv import load_dotenv





#response = model.generate_content("What is the meaning of life?")
#print(response.text)

#load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))


# Now proceed with deserialization
# For example:
# result = genai.deserialize_from_pickle(pickle_file)


def get_pdf_text(pdf_docs):
    text = ""
    # iterate over all pdf files uploaded
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # iterate over all pages in a pdf
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # create an object of RecursiveCharacterTextSplitter with specific chunk size and overlap size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    # now split the text we have using object created
    chunks = text_splitter.split_text(text)

    return chunks

    
def make_prompt(relevant_passage):
    escaped = relevant_passage[0].replace("'", "").replace('"', "").replace("\n", " ")
    query = """can you analyse the given question papers and find out the important topics based on frequently asked questions maximum of
            7-9 topics can be listed as important topics and if their are more than 10 topics then high mark distribution topics can be selected
            others topics can be neglected. The output should only contain the important topics no other sentences must be included.
            this is for students to help in their preparation of exams .the students will have an average understanding of the topic
            so no explanation of topic is needed.please make sure to include the ones mentioned in question paper even if the improtant
            topics is below 5 its no problem in that case include only that many.ease exclude (if applicable to your specific course) topics"""
    prompt =(""" you are a helpful prompt which helps students to find the important topics for exam from uploaded pdfs. outputs must be concise.
    you are like the support mentor helping students finding important topics based on repeated questions.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer
    


def main():
    st.set_page_config("Help-Ed")
    st.image("logo.png", width=200)  # Add the logo image
    st.header("Navigate Your Learning With Us!!")

    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            #get_vector_store(text_chunks)
            answer = make_prompt(text_chunks)
            st.success("Done")

        # Display the answer below the header
        st.write("Reply: ", answer.text)


if __name__ == "__main__":
    main()

