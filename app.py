import os
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return processor, blip_model, embed_model


processor, blip_model, embed_model = load_models()

st.title("Image RAG Question Answering")

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # Generate caption (hidden)
    # -----------------------------
    inputs = processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # -----------------------------
    # Store in vector DB
    # -----------------------------
    doc_embedding = embed_model.encode([caption])
    dimension = doc_embedding.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embedding))

    # -----------------------------
    # Ask question
    # -----------------------------
    question = st.text_input("Ask a question about the image:")

    if question:
        query_embedding = embed_model.encode([question])
        D, I = index.search(np.array(query_embedding), k=1)

        retrieved_text = caption  # internal use only

        answer = f"Answer: {retrieved_text}"
        st.write(answer)