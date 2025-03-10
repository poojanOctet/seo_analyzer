import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import wikipedia
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv
import logging, psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample prompt:
    # Requirements:
    # - Title under 60 chars with keyword 
    # - Meta description 150-160 chars with keyword
    # - {request.length} word article
    # - Keyword density 1-2%
    # - Use H2 headers & H3 headers
    # - Output format:
    # Title: [title]
    # Meta: [meta description]
    # Content: [content]"""

load_dotenv()

class OpenAIEmbedder:
    def __init__(self, client, model="text-embedding-3-small"):
        self.client = client
        self.model = model

    def encode(self, text):
        if isinstance(text, str):
            input_list = [text]
        else:
            input_list = text
        response = self.client.embeddings.create(
            input=input_list,
            model=self.model
        )

        return [data.embedding for data in response.data]

# Initialize models and clients
# EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

EMBEDDER = OpenAIEmbedder(client)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/static", StaticFiles(directory="app/static"), name="static")
# templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def home():
    # return templates.TemplateResponse("index.html", {"request": request})
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory used: {memory_info.rss / 1024 / 1024:.2f} MB")
    return {"message": "Welcome to the SEO Content Generator API!"}

class ContentRequest(BaseModel):
    topic: str
    keyword: str
    length: int = 1000

class ContentResponse(BaseModel):
    content: str

class CommandRequest(BaseModel):
    content: str

def get_wikipedia_content(topic: str):
    try:
        return wikipedia.page(topic).content
    except:
        return wikipedia.summary(topic)

def split_text(text: str, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@app.post("/generate-content")
async def generate_content(request: ContentRequest):
    # Knowledge Base Setup
    article_content = get_wikipedia_content(request.topic)
    chunks = split_text(article_content)
    
    # Create and store embeddings
    embeddings = EMBEDDER.encode(chunks)
    vector_store = FAISS.from_embeddings(list(zip(chunks, embeddings)), EMBEDDER)
    
    # Retrieval Mechanism
    query_embedding = EMBEDDER.encode([request.topic])
    retrieved_docs = vector_store.similarity_search_by_vector(query_embedding[0], k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Content Generation with SEO optimization
    prompt = f"""Generate an SEO-optimized article about {request.topic} targeting the keyword '{request.keyword}'.
    Use this context: {context}
    
    Requirements:
    - Title under 60 chars with keyword and it should be action-oriented or more-compelling
    - Meta description 150-160 chars with keyword which is enticing
    - {request.length} word article 
    - Keyword density 1-2%
    - Use H2 headers & H3 headers
    - Use bullet points
    - Use variation in keyword phrasing
    - mention external links (2 to 3) to the {request.topic}
    - Output format:
    Title: [title]
    Meta: [meta description]
    Content: [content]"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory used: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    return parse_response(response.choices[0].message.content)

def parse_response(text: str):
    # Simple response parser
    parts = text.split("\n\n")
    return {
        "title": parts[0].replace("Title: ", ""),
        "meta": parts[1].replace("Meta: ", ""),
        "content": "\n\n".join(parts[2:])
    }

@app.post("/add-more", response_model=ContentResponse)
async def add_more(request: CommandRequest):
    content = request.content
    prompt = f"Add 30% more SEO optimized content to this (DO NOT USE ANY PREFIX LIKE 'Here is 30% more content'): {content}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"content": response.choices[0].message.content}

@app.post("/paraphrase", response_model=ContentResponse)
async def paraphrase(request: CommandRequest):
    content = request.content
    prompt = f"Paraphrase this content to be SEO optimized (DO NOT USE ANY PREFIX LIKE 'Here is paraphrased version'): {content}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"content": response.choices[0].message.content}

@app.post("/improve", response_model=ContentResponse)
async def improve(request: CommandRequest):
    content = request.content
    prompt = f"Improve this content to be SEO optimized (DO NOT USE ANY PREFIX LIKE 'Here is improved version'): {content}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"content": response.choices[0].message.content}

@app.post("/summarize", response_model=ContentResponse)
async def summarize(request: CommandRequest):
    content = request.content
    prompt = f"Summarize this content (DO NOT MENTION ANY PREFIX LIKE 'Here is the summary'): {content}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"content": response.choices[0].message.content}

@app.post("/write-analogy", response_model=ContentResponse)
async def write_analogy(request: CommandRequest):
    content = request.content
    prompt = f"Write appropriate analogy for this content (DO NOT MENTION ANY PREFIX LIKE 'Sure! Here’s an analogy that captures the essence of the content:'): {content}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"content": response.choices[0].message.content}

@app.post("/fix-grammar", response_model=ContentResponse)
async def fix_grammer(request: CommandRequest):
    content = request.content
    prompt = f"Fix grammar of this content, do not loose the sense of the content & do not use any prefix like 'Sure! Here is corrected sentence': {content}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"content": response.choices[0].message.content}