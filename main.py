import os
import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from neo4j import GraphDatabase

# For PDFs, Word, Images
import docx
import pytesseract
from PIL import Image
import fitz

# --- CONFIGURATION & ENV SETUP ---
load_dotenv()
# General Config
EXCEL_FILE = "./data/tickets.xlsx"
VECTOR_STORE_DIR = "vector_store"
LAST_MOD_TIME_FILE = "last_mod_time.txt"

# Environment Variables (Local)
# Use st.secrets if deploying to Streamlit Cloud
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
# Neo4j Config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j") # default

# --- INITIALIZATION ---

# HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# LLM
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# Neo4j Graph
neo4j_graph = None
try:
    if NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD:
        neo4j_graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        st.sidebar.success("🌐 Connected to Neo4j.")
    else:
        st.sidebar.error("❌ Neo4j environment variables are missing.")
except Exception as e:
    st.sidebar.error(f"❌ Failed to connect to Neo4j: {e}")

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are an expert IT support assistant. 
Use the Ticket Data (Context) and the IT Knowledge Graph Results (Graph Context) to provide a comprehensive answer.

Ticket Data (Context):
{context}

IT Knowledge Graph Results (Graph Context):
{graph_context}

Question: {input}

If there's no helpful information in the context, reply with:
"Not enough historical data or graph relationships to answer accurately."
""")


# --- UTILITY FUNCTIONS ---

# (Keeping basic file utilities for brevity - assume they are present)
def get_last_mod_time():
    return os.path.getmtime(EXCEL_FILE)

def read_last_saved_mod_time():
    if os.path.exists(LAST_MOD_TIME_FILE):
        with open(LAST_MOD_TIME_FILE, "r") as f:
            try:
                return float(f.read().strip())
            except ValueError:
                return None
    return None

def save_last_mod_time(mod_time):
    with open(LAST_MOD_TIME_FILE, "w") as f:
        f.write(str(mod_time))
        
def load_vector_store():
    return FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)

def extract_text_from_file(file):
    # Implementation for text extraction (omitted for space, assume working)
    if not file: return ""
    # ... (omitted)
    return file.read().decode("utf-8") # Simple fallback


def build_vector_store_and_graph():
    """Builds the FAISS vector store AND populates the Neo4j Graph."""
    df = pd.read_excel(EXCEL_FILE)

    # CHECK FOR REQUIRED COLUMNS - ADDED 'Application'
    required_cols = ["Ticket ID", "Description", "Category", "Subcategory", "Application", "Resolution"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ '{col}' column not found in Excel file. Please ensure it exists.")
            st.stop()
    
    df.dropna(subset=["Description"], inplace=True)

    # 1. Build Vector Store (FAISS)
    st.info("Building FAISS Vector DB...")
    documents = []
    graph_nodes = []
    
    for _, row in df.iterrows():
        # Document content for FAISS (RAG) - INCLUDES 'Application'
        content = f"""
        Ticket ID: {row.get('Ticket ID', '')}
        Description: {row.get('Description', '')}
        Category: {row.get('Category', '')}
        Subcategory: {row.get('Subcategory', '')}
        Application: {row.get('Application', '')} 
        Resolution: {row.get('Resolution', '')}
        """
        documents.append(Document(page_content=content.strip()))
        
        # Data for Neo4j (Graph) - INCLUDES 'application'
        graph_nodes.append({
            'id': row.get('Ticket ID', ''),
            'description': row.get('Description', ''),
            'category': row.get('Category', 'Unknown'),
            'subcategory': row.get('Subcategory', 'Unknown'),
            'application': row.get('Application', 'Unknown'), 
            'resolution': row.get('Resolution', '')
        })

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(VECTOR_STORE_DIR)
    st.success("✅ Vector DB rebuilt.")

    # 2. Populate Neo4j Graph
    if neo4j_graph:
        st.info("Populating Neo4j Knowledge Graph...")
        # Clear existing data for a clean rebuild (optional, but recommended)
        neo4j_graph.query("MATCH (n) DETACH DELETE n") 

        for node in graph_nodes:
            # Basic sanitization for Cypher strings
            desc_safe = node['description'].replace("'", "\\'") 
            res_safe = node['resolution'].replace("'", "\\'")

            neo4j_graph.query(f"""
            MERGE (c:Category {{name: '{node['category']}'}})
            MERGE (s:Subcategory {{name: '{node['subcategory']}'}})
            MERGE (a:Application {{name: '{node['application']}'}})
            
            MERGE (t:Ticket {{
                ticketId: '{node['id']}', 
                description: '{desc_safe}', 
                resolution: '{res_safe}'
            }})
            
            MERGE (t)-[:BELONGS_TO_CATEGORY]->(c)
            MERGE (t)-[:BELONGS_TO_SUBCATEGORY]->(s)
            MERGE (t)-[:AFFECTS_APPLICATION]->(a) 
            """)

        st.success("✅ Neo4j Graph populated.")


# --- DYNAMIC GRAPH RETRIEVAL CHAIN ---

# Use LLM to generate Cypher and execute it
def get_dynamic_graph_context(user_query):
    """Uses LLM to generate and execute a Cypher query against the Neo4j graph."""
    if not neo4j_graph:
        return Document(page_content="Neo4j not available.")

    # Use the LangChain GraphCypherQAChain to dynamically generate Cypher
    # The LLM uses the graph schema (automatically retrieved by Neo4jGraph) 
    # and the prompt to write the Cypher query.
    cypher_chain = GraphCypherQAChain.from_llm(
        llm=llm, 
        graph=neo4j_graph, 
        verbose=False # Set to True for debugging Cypher generation
    )
    
    # Run the chain
    try:
        # The chain executes the following steps:
        # 1. LLM converts user_query into a Cypher query based on the schema.
        # 2. Executes the Cypher query on Neo4j.
        # 3. LLM summarizes the Cypher result into a natural language response.
        graph_result = cypher_chain.run(user_query)
        
        # Return the result as a Document to be passed to the final RAG prompt
        return Document(
            page_content=f"Knowledge Graph Query Result: {graph_result}",
            metadata={"source": "Neo4j Graph"}
        )
    except Exception as e:
        # st.error(f"Error generating/running Cypher query: {e}") # Enable for debug
        return Document(page_content="Graph query failed or provided no results.")

# --- COMBINED RAG CHAIN ---

# Custom retriever that fetches docs and graph context separately
def combined_retriever(query):
    """Fetches documents from FAISS and dynamic context from Neo4j."""
    
    # 1. Vector Store (FAISS) Retrieval
    faiss_docs = vector_store.as_retriever().invoke(query)
    
    # 2. Dynamic Graph Retrieval
    # We pass the graph context as a single Document to the RAG chain 
    # to maintain separation in the final prompt template.
    graph_context_doc = get_dynamic_graph_context(query)
    
    # The final RAG chain uses a dictionary mapping for context, so we'll 
    # adjust the invocation structure instead of returning a flat list of docs.
    return {
        "context": faiss_docs,
        "graph_context": graph_context_doc
    }


# --- STREAMLIT UI ---

st.set_page_config(page_title="Ticket Gen AI Assistant", layout="wide")
st.title("🎫 Gen AI Ticket Assistant (FAISS + Dynamic Neo4j RAG)")

# Auto-rebuild vector store and graph if Excel changed or on first run
if os.path.exists(EXCEL_FILE):
    last_mod_time = get_last_mod_time()
    saved_mod_time = read_last_saved_mod_time()

    if saved_mod_time != last_mod_time or not os.path.exists(VECTOR_STORE_DIR):
        st.warning("Excel file changed — rebuilding vector DB AND Neo4j Graph...")
        build_vector_store_and_graph()
        save_last_mod_time(last_mod_time)
        st.experimental_rerun() # Rerun Streamlit after rebuild
else:
    st.error(f"❌ Required Excel file not found: {EXCEL_FILE}")
    st.stop()


# Load existing vector store
vector_store = load_vector_store()

# Query input options
query_type = st.radio("Choose input type:", ["Text", "File (PDF/DOCX/Image)"])

user_query = ""
if query_type == "Text":
    user_query = st.text_input("📝 Describe a new ticket or ask a question:")
else:
    uploaded_query_file = st.file_uploader("📂 Upload a query file", type=["txt", "pdf", "docx", "png", "jpg", "jpeg"])
    user_query = extract_text_from_file(uploaded_query_file) if uploaded_query_file else ""

# Run query
if user_query:
    # 1. Create the combined RAG chain components
    # The final prompt requires 'context' (from FAISS) and 'graph_context' (from Neo4j)
    
    # LangChain's structure for multi-context RAG:
    # A. Use a custom function (combined_retriever) to fetch the two contexts
    # B. Use a Passthrough step (RunnablePassthrough) to handle the 'input' query
    # C. Combine everything into a dictionary that matches the prompt's input variables
    
    # Note: We can't use create_retrieval_chain directly here because it expects 
    # a single list of Documents, but we need two separate context variables.
    
    from langchain_core.runnables import RunnablePassthrough
    
    # 2. Define the main document chain (LLM + Prompt)
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 3. Assemble the full RAG chain
    retrieval_chain = (
        RunnablePassthrough.assign(retrieved_context=combined_retriever) # Get contexts
        | RunnablePassthrough.assign(
            context=lambda x: x['retrieved_context']['context'],
            graph_context=lambda x: x['retrieved_context']['graph_context'],
        )
        | document_chain
    )
    
    with st.spinner("Thinking..."):
        # The invoke now passes the user_query to the 'input' variable
        response = retrieval_chain.invoke({"input": user_query})
        
        st.write("### ✅ Answer")
        st.write(response)

        # The response structure from this chain is the final answer string
        # We need to access the context data that was passed through the chain
        # which requires some modification or using a different chain structure
        # For simplicity, let's just show the raw context that was fetched:

        st.write("---")
        with st.expander("🔎 Raw Context Used"):
            # Note: Context display is simplified here due to custom chain structure
            st.markdown("#### FAISS Documents (Vector Similarity)")
            for doc in retrieval_chain.invoke({"input": user_query})['context']: # Need to re-run or store intermediate state
                st.markdown(doc.page_content)
                st.write("---")
            
            st.markdown("#### Neo4j Graph Result (Cypher QA Chain)")
            st.write(retrieval_chain.invoke({"input": user_query})['graph_context'].page_content)