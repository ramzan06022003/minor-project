import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI   
from langchain.prompts import PromptTemplate # Note the import change
from langchain.chains import RetrievalQA # The key import for the legacy approach
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from linkedin_api import Linkedin
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# --- STEP 1: THE SCRAPER MODULE (Unchanged) ---
def scrape_linkedin_profile(profile_url: str) -> str:
    mail="tejas.jangra2001@gmail.com"
    Linkedin_username = "Tejas Jangra"
    Linkedin_url = "https://www.linkedin.com/in/tejas-jangra"
    # """
    # Simulates scraping a LinkedIn profile.
    # In a real application, this would use an API to fetch the data.
    # """
    # print(f"INFO: Simulating scraping for URL: {profile_url}")

    # return """
    # Priya Sharma | Senior AI/ML Engineer @ QuantumLeap Inc.

    # Summary:
    # A seasoned AI/ML Engineer with over 8 years of experience in architecting and deploying scalable machine learning solutions.
    # Specializing in Natural Language Processing (NLP) and computer vision, with a proven track record of leading cross-functional
    # teams to deliver high-impact projects. My core competency lies in the full project lifecycle, from ideation and data
    # pipelining to model deployment and post-launch optimization. Passionate about leveraging cutting-edge deep learning
    # frameworks to solve complex business problems and drive innovation.

    # Experience:
    # 1. Senior AI/ML Engineer, QuantumLeap Inc. (2020 - Present)
    #    - Led the development of a proprietary NLP engine that improved customer sentiment analysis accuracy by 35%.
    #    - Deployed a real-time computer vision system on edge devices for manufacturing defect detection, reducing waste by 15%.
    #    - Engineered a multi-modal recommendation system using TensorFlow and PyTorch, resulting in a 20% uplift in user engagement.
    #    - Mentored a team of 4 junior engineers, fostering a culture of technical excellence and continuous learning.

    # 2. Machine Learning Engineer, DataDriven Solutions (2017 - 2020)
    #    - Developed predictive maintenance models for industrial IoT devices, preventing critical failures and saving an estimated $2M annually.
    #    - Implemented scalable data processing pipelines using Apache Spark and Kafka for handling terabytes of streaming data.

    # Skills:
    # - Languages: Python, C++, Java
    # - ML/DL Frameworks: TensorFlow, PyTorch, Scikit-learn, Keras
    # - MLOps: Docker, Kubernetes, Kubeflow, MLflow
    # - Big Data: Apache Spark, Hadoop, Kafka
    # - Cloud: AWS, Google Cloud Platform (GCP)
    # """
    api = Linkedin(mail, os.getenv("pass"))
    linkedin_profile =api.get_profile("tejas-jangra")
    return linkedin_profile

# --- STEP 2 & 3: PROCESSING & INGESTION (Unchanged) ---
def ingest_and_create_retriever(profile_text: str):
    """
    Takes raw text, splits it, embeds it, and stores it in ChromaDB.
    Returns a retriever object.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content=profile_text)]
    chunks = text_splitter.split_documents(docs)
    print(f"INFO: Split profile into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("INFO: Initialized OpenAI embeddings model.")

    vectorstore = Chroma.from_documents(chunks, embeddings)
    print("INFO: Ingested chunks into ChromaDB.")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )
    print("INFO: Created retriever from vector store.")
    return retriever

# --- STEP 4: RETRIEVAL, AUGMENTATION & GENERATION (MODIFIED) ---
def setup_rag_chain(retriever):
    """
    Sets up the RAG chain using the legacy RetrievalQA class.
    """
    # 1. Define the prompt template for the LLM
    template = """
    As a world-class Talent Acquisition Strategist AI, your task is to synthesize a professional analysis
    based on the provided context from a candidate's profile. Do not mention that you are an AI.
    Adopt a highly professional, technically sophisticated, and slightly jargon-heavy tone.

    Your analysis should be concise, insightful, and highlight key vectors of expertise, potential synergies,
    and strategic value.

    Context from the profile:
    {context}

    Based on the context, answer the following question:
    Question: {question}

    Analysis:
    """
    # Note: Using PromptTemplate instead of ChatPromptTemplate for compatibility with some legacy chains
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 2. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model_name="gemini-2.5-flash", temperature=0.3, api_key=os.getenv("GOOGLE_API_KEY"))

    # 3. Create the RetrievalQA chain
    # This chain type 'stuffs' the retrieved documents into the context of the prompt.
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True # Optional: to see which chunks were retrieved
    )
    print("INFO: RetrievalQA chain is configured and ready.")
    return rag_chain


# --- MAIN EXECUTION BLOCK (MODIFIED) ---
if __name__ == "__main__":
    linkedin_url = "https://www.linkedin.com/in/priya-sharma-example"

    # Execute the pipeline
    raw_profile_data = scrape_linkedin_profile(linkedin_url)
    # profile_retriever = ingest_and_create_retriever(raw_profile_data)
    # analysis_chain = setup_rag_chain(profile_retriever)

    # 🚀 Ask a question to the RAG system
    print("\n" + "="*50)
    query = "Evaluate her leadership and project impact from a senior stakeholder's perspective."
    print(f"Executing query: {query}\n")

    # The legacy chain expects the query in a dictionary and returns a dictionary.
    # result = analysis_chain({"query": query})

    # print("✅ Final Analysis:\n")
    # # The final text answer is in the 'result' key
    # print(result['result'])
    # You can also inspect the source documents that the retriever found
    # print("\n--- Source Documents ---")
    # for doc in result['source_documents']:
    #     print(doc.page_content)
    #     print("-" * 20)
    # print("="*50 + "\n")

    # # Example 2
    # query_2 = "What is her core technical stack and how would it integrate with our MLOps platform?"
    # print(f"Executing query: {query_2}\n")
    # result_2 = analysis_chain({"query": query_2})
    # print("✅ Final Analysis:\n")
    # print(result_2['result'])
    # print("="*50)

    print(raw_profile_data)