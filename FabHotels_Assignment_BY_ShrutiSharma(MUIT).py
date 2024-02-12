#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx
import PyPDF2
import pdfplumber


# In[2]:


# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

def preprocess_resume(text):
    return text


# In[3]:


def extract_entities(text):
    doc = nlp(text)
    entities = {
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": []
    }
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            entities["skills"].append(ent.text)
        elif ent.label_ == "DATE" and "experience" in ent.text.lower():
            entities["experience"].append(ent.text)
        elif ent.label_ == "EDUCATION":
            entities["education"].append(ent.text)
        elif ent.label_ == "CERTIFICATION":
            entities["certifications"].append(ent.text)
    return entities


# In[4]:


def rank_resumes(resumes, job_description):
    job_description = preprocess_resume(job_description)
    vectorizer = TfidfVectorizer()
    job_vector = vectorizer.fit_transform([job_description])

    ranked_resumes = []
    for resume in resumes:
        resume = preprocess_resume(resume)
        resume_vector = vectorizer.transform([resume])
        similarity_score = cosine_similarity(job_vector, resume_vector)[0][0]
        ranked_resumes.append((resume, similarity_score))

    ranked_resumes.sort(key=lambda x: x[1], reverse=True)
    return ranked_resumes


# In[5]:


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# In[6]:


def read_docx(file_path):
    doc = docx.Document(file_path)
    text = [para.text for para in doc.paragraphs]
    return '\n'.join(text)


# In[7]:


def read_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfFileReader(f)
        text = [reader.getPage(page_num).extractText() for page_num in range(reader.numPages)]
        return '\n'.join(text)


# In[8]:


def read_pdfplumber(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = [page.extract_text() for page in pdf.pages]
        return '\n'.join(text)


# In[9]:


def read_resume(file_path):
    if file_path.endswith('.txt'):
        return read_text_file(file_path)
    elif file_path.endswith('.docx'):
        return read_docx(file_path)
    elif file_path.endswith('.pdf'):
        try:
            return read_pdf(file_path)
        except:
            return read_pdfplumber(file_path)
    else:
        raise ValueError("Unsupported file format")


# In[10]:


job_description = input()


# In[11]:


filed=input()


# In[ ]:


ranked_resumes = rank_resumes(resumes, job_description)


# In[ ]:


for idx, (resume, score) in enumerate(ranked_resumes, start=1):
    print(f"Rank {idx}: Similarity Score: {score:.2f}\n{resume}\n")


# In[ ]:




