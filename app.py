# ============================
# RAG Cloud Agent – FAISS Edition (Single File)
# ============================
# - Constants consolidated at the top
# - FAISS local vector store (save/load)
# - Preserves prior features (CFN/Ansible/Python generation & execution)
# - Exact-match reuse via FAISS similarity
# - Editor + logs + chat history + related chunks

# ---------- CONSTANTS ----------
APP_TITLE = "RAG Cloud Agent (FAISS-Powered)"
DEFAULT_REGION = "us-east-1"

# Embeddings / Vector DB

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
# EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


EMBEDDING_DEVICE = "cpu"
FAISS_DIR = "vector_index/faiss"  # folder to save/load FAISS index

# LLM (Ollama)
# OLLAMA_MODEL = "mistral:latest"
OLLAMA_MODEL = "llama3.2:latest"

# Chunking / Retrieval defaults (editable in UI)
CHUNK_SIZE_DEFAULT = 4000
CHUNK_OVERLAP_DEFAULT = 100
TOP_K_DEFAULT = 4
SIM_THRESHOLD_DEFAULT = 0.95  # similarity threshold for "reuse from DB"

# Script folders
DIR_TERRAFORM = "scripts/terra"
DIR_CFN = "scripts/cfn"
DIR_ANSIBLE = "scripts/ansi"
DIR_PYTHON = "scripts/others"  # per your requirement
DIR_TEXT = "scripts/other"

# ---------- IMPORTS ----------
import sys
import types
import os
import tempfile
import git
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import boto3
from botocore.exceptions import ClientError

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, PyPDFium2Loader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import requests
from urllib.parse import urljoin
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ---------- Torch Fix (some HF envs need this) ----------
torch_classes_dummy = types.ModuleType("torch.classes")
torch_classes_dummy.__path__ = []
sys.modules["torch.classes"] = torch_classes_dummy

# ---------- UI / SESSION ----------
st.title(APP_TITLE)

for key in [
    "retriever", "vectorstore", "generated_script", "responses",
    "documents", "chunks", "history", "logs"
]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["responses", "documents", "chunks", "history", "logs"] else None

# Sidebar controls
option = st.sidebar.selectbox("Choose input method", ["Upload PDFs", "Enter URLs"])
chunk_size = st.sidebar.slider("Chunk size (chars)", 1000, 8000, CHUNK_SIZE_DEFAULT, 500)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 800, CHUNK_OVERLAP_DEFAULT, 50)
top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 10, TOP_K_DEFAULT)
sim_threshold = st.sidebar.slider("Similarity threshold (reuse from FAISS)", 0.50, 0.99, SIM_THRESHOLD_DEFAULT, 0.01)

# ---------- UTILS ----------
def is_github_url(url: str) -> bool:
    return "github.com" in url and not url.endswith(".pdf")

def load_mkdocs_site(base_url: str):
    """
    Loads ALL MkDocs pages by reading the Material search index.
    Falls back to loading just the base URL if index not found.
    """
    base_url = base_url.rstrip("/") + "/"
    idx_url = urljoin(base_url, "search/search_index.json")
    docs = []

    try:
        j = requests.get(idx_url, timeout=10).json()
        # Material’s payload has either top-level "docs" or nested under "index"
        items = j.get("docs") or j.get("index", {}).get("docs") or []
        # Each item has "location" (page path, maybe with fragment), "title", "text"
        urls = []
        seen = set()
        for it in items:
            loc = it.get("location", "")
            # strip any #fragment to avoid duplicates
            page_url = urljoin(base_url, loc.split("#", 1)[0])
            if page_url not in seen and page_url.startswith(base_url):
                urls.append(page_url)
                seen.add(page_url)

        if not urls:
            # Fallback: at least load the base page
            urls = [base_url]

        # Use WebBaseLoader on the final set (deduped)
        loader = WebBaseLoader(urls)
        docs = loader.load()

    except Exception as e:
        # Fallback to just the base page
        docs = WebBaseLoader([base_url]).load()

    return docs

def load_github_repo(url: str) -> List[Document]:
    documents = []
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, "repo")
        try:
            git.Repo.clone_from(url, repo_path)
            for filepath in Path(repo_path).rglob("*"):
                if filepath.is_file() and filepath.suffix.lower() in [".py", ".md", ".txt", ".yaml", ".yml", ".json"]:
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        documents.append(Document(page_content=content, metadata={"source": str(filepath)}))
                    except Exception as e:
                        st.warning(f"Skipped {filepath} due to error: {e}")
        except Exception as e:
            st.error(f"Failed to clone repository: {e}")
    return documents

def extract_region_from_code(code: str, default_region: str = DEFAULT_REGION) -> str:
    match = re.search(r"region\s*[:=]\s*['\"]([\w-]+)['\"]", code, re.IGNORECASE)
    return match.group(1) if match else default_region

def sanitize_cloudformation_template(yaml_text: str) -> str:
    # Drop fields that break CFN validate/apply in some exports
    skip_fields = ["CreationPreconditions"]
    return "\n".join(line for line in yaml_text.splitlines() if not any(f in line for f in skip_fields))

def deploy_cloudformation(template_body: str, stack_name: str, region: str = DEFAULT_REGION) -> List[str]:
    logs = []
    try:
        session = boto3.Session(region_name=region)
        creds = session.get_credentials()
        if creds:
            fc = creds.get_frozen_credentials()
            logs.append(f"**AWS Credentials in Use:**\n- Access Key: {fc.access_key}\n- Region: {region}")
        cf = session.client("cloudformation")
        cf.validate_template(TemplateBody=template_body)
        logs.append("✅ Template validated successfully.")
        resp = cf.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"],
        )
        logs.append(f"🚀 Stack creation started. StackId: {resp['StackId']}")
        waiter = cf.get_waiter("stack_create_complete")
        waiter.wait(StackName=stack_name)
        logs.append("✅ Stack creation completed successfully.")
    except ClientError as e:
        logs.append(f"❌ AWS ClientError: {e.response['Error'].get('Message', str(e))}")
    except Exception as e:
        logs.append(f"❌ Error: {str(e)}")
    return logs

def deploy_ansible_playbook(path_to_playbook: str) -> List[str]:
    logs = [f"🚀 Running Ansible playbook: {path_to_playbook}"]
    try:
        result = subprocess.run(["ansible-playbook", path_to_playbook], capture_output=True, text=True)
        logs.append("🔧 STDOUT:\n" + result.stdout)
        if result.stderr:
            logs.append("⚠️ STDERR:\n" + result.stderr)
        logs.append("✅ Success." if result.returncode == 0 else f"❌ Exit {result.returncode}")
    except FileNotFoundError:
        logs.append("❌ 'ansible-playbook' not found. Ensure Ansible is installed and on PATH.")
    except Exception as e:
        logs.append(f"❌ Ansible Error: {str(e)}")
    return logs

def run_python_script(script_path: str) -> List[str]:
    logs = [f"▶️ Running Python script: {script_path}"]
    try:
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        logs.append("▶️ STDOUT:\n" + result.stdout)
        if result.stderr:
            logs.append("⚠️ STDERR:\n" + result.stderr)
        logs.append("✅ Script ran successfully." if result.returncode == 0 else f"❌ Exit {result.returncode}")
    except Exception as e:
        logs.append(f"❌ Python run error: {str(e)}")
    return logs

def classify_artifact(text: str) -> str:
    t = text.lower()
    if "awstemplateformatversion" in t or "aws::" in t:
        return "cfn"
    if "hosts:" in t or "ansible" in t:
        return "ansible"
    if "terraform" in t and ("resource" in t or "provider" in t):
        return "terraform"
    if "import boto3" in t or "boto3." in t:
        return "python"
    return "text"

def detect_intent(query: str) -> Dict[str, Any]:
    q = query.lower()
    action = "answer"
    if any(k in q for k in ["create", "provision", "spin up", "launch"]):
        action = "create"
    elif any(k in q for k in ["update", "modify", "change"]):
        action = "update"
    elif any(k in q for k in ["delete", "destroy", "tear down", "terminate"]):
        action = "delete"
    elif any(k in q for k in ["show", "list", "describe", "get"]):
        action = "describe"

    resources = []
    for key, aliases in {
        "s3": ["s3", "bucket"],
        "rds": ["rds", "postgres", "mysql", "database"],
        "ec2": ["ec2", "instance", "vm"],
        "vpc": ["vpc", "network"],
        "subnet": ["subnet"],
        "sg": ["security group", "sg"],
        "iam": ["iam", "role", "policy"],
    }.items():
        if any(a in q for a in aliases):
            resources.append(key)

    artifact_hint = "auto"
    if "ansible" in q:
        artifact_hint = "ansible"
    elif "cloudformation" in q or "cfn" in q:
        artifact_hint = "cfn"
    elif "terraform" in q:
        artifact_hint = "terraform"
    elif "python" in q or "boto3" in q:
        artifact_hint = "python"

    return {"action": action, "resources": resources or ["generic"], "artifact_hint": artifact_hint}

RESOURCE_ORDER = ["vpc", "subnet", "sg", "iam", "s3", "rds", "ec2"]

def plan_steps(intent: Dict[str, Any]) -> List[str]:
    if intent["action"] != "create":
        return intent["resources"]
    wanted = [r for r in RESOURCE_ORDER if r in intent["resources"]] or intent["resources"]
    return wanted

def load_pdf_robust(path: str):
    """
    Try PDFium first (most robust), then fall back to PyPDF.
    Add more fallbacks if you like (PyMuPDF, Unstructured, etc.)
    """
    # 1) PDFium (recommended)
    try:
        return PyPDFium2Loader(path).load()
    except Exception as e1:
        st.warning(f"PDFium parse failed: {e1}. Falling back to PyPDF…")
    # 2) PyPDF (what you currently use)
    try:
        return PyPDFLoader(path).load()
    except Exception as e2:
        st.error(f"PyPDF parse failed: {e2}")
        return []


# ---------- INGESTION ----------
if option == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        documents: List[Document] = []
        for uf in uploaded_files:
            with open("/tmp/_temp.pdf", "wb") as tf:
                tf.write(uf.read())
            # documents.extend(PyPDFLoader("/tmp/_temp.pdf").load())
            documents.extend(load_pdf_robust("/tmp/_temp.pdf"))
        from langchain.text_splitter import CharacterTextSplitter
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)
        st.session_state.documents, st.session_state.chunks = documents, chunks
        st.write(f"Loaded {len(documents)} docs → {len(chunks)} chunks")

elif option == "Enter URLs":
    urls = st.text_area("Enter URLs (one per line)").splitlines()
    if urls:
        documents: List[Document] = []
        for url in urls:
            if is_github_url(url):
                documents.extend(load_github_repo(url))
            else:
                try:
                    # If they give the base of an MkDocs site, crawl all pages; else load the URL as-is
                    if url.rstrip("/").endswith(("127.0.0.1:8000", "localhost:8000")) or url.endswith("/"):
                        documents.extend(load_mkdocs_site(url))
                    else:
                        documents.extend(WebBaseLoader([url]).load())
                except Exception as e:
                    st.warning(f"Failed to load {url}: {e}")
        from langchain.text_splitter import CharacterTextSplitter
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)
        st.session_state.documents, st.session_state.chunks = documents, chunks
        st.write(f"Loaded {len(documents)} → {len(chunks)} chunks")

# ---------- FAISS & EMBEDDINGS ----------
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={
        "device": EMBEDDING_DEVICE,
        "trust_remote_code": True  
    },
    encode_kwargs={"normalize_embeddings": True}
)

def build_faiss_index(docs: List[Document], save_dir: str) -> FAISS:
    vs = FAISS.from_documents(docs, embeddings)
    os.makedirs(save_dir, exist_ok=True)
    vs.save_local(save_dir)
    return vs

def load_faiss_index(load_dir: str) -> FAISS:
    if not os.path.isdir(load_dir):
        raise FileNotFoundError(f"FAISS index directory not found: {load_dir}")
    # allow_dangerous_deserialization=True is required due to pickled docstore
    return FAISS.load_local(load_dir, embeddings, allow_dangerous_deserialization=True)

col_build, col_load = st.columns(2)
with col_build:
    if st.button("Build FAISS Index"):
        if st.session_state.chunks:
            st.session_state.vectorstore = build_faiss_index(st.session_state.chunks, FAISS_DIR)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
            st.success(f"✅ Built & saved FAISS index → {FAISS_DIR}")
        else:
            st.warning("No chunks to embed.")

with col_load:
    if st.button("Load FAISS Index"):
        try:
            st.session_state.vectorstore = load_faiss_index(FAISS_DIR)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
            st.success(f"✅ Loaded FAISS index from {FAISS_DIR}")
        except Exception as e:
            st.error(f"❌ Error loading FAISS: {e}")

# ---------- LLM & ORCHESTRATION ----------
llm = OllamaLLM(model=OLLAMA_MODEL)
user_query = st.text_input("Enter your query:")
generate = st.button("Generate")

ctx_box = st.expander("Retrieved context (top-K)", expanded=False)

def faiss_topk_with_scores(query: str, k: int) -> List[Tuple[Document, float]]:
    """
    FAISS.similarity_search_with_score returns (doc, distance) where lower is better.
    We map to a similarity in [0,1] via 1 / (1 + distance) for an intuitive threshold.
    """
    if not st.session_state.vectorstore:
        return []
    pairs = st.session_state.vectorstore.similarity_search_with_score(query, k=k)
    out = []
    for doc, distance in pairs:
        similarity = 1.0 / (1.0 + float(distance))
        doc.metadata = dict(doc.metadata or {})
        doc.metadata["distance"] = float(distance)
        doc.metadata["similarity"] = float(similarity)
        out.append((doc, similarity))
    return out

if generate and user_query and st.session_state.retriever:
    # 1) Intent & plan
    intent = detect_intent(user_query)
    steps = plan_steps(intent)

    # 2) Retrieve context FIRST (source of truth)
    results_scored = faiss_topk_with_scores(user_query, top_k)
    results = [d for (d, _) in results_scored]

    with ctx_box:
        for i, doc in enumerate(results, 1):
            sim = doc.metadata.get("similarity", 0.0)
            st.markdown(f"**Chunk {i} (sim≈{sim:.3f})**\n\n````\n{doc.page_content[:1200]}\n````")

    # Save history (newest first)
    st.session_state.history.insert(0, {
        "query": user_query,
        "intent": intent,
        "chunks": [r.page_content[:400] for r in results]
    })

    # 3) Try exact reuse from FAISS
    if results_scored:
        top_doc, top_sim = results_scored[0]
        if top_sim >= sim_threshold:
            st.session_state.generated_script = top_doc.page_content
            st.success(f"✅ Reused high-similarity match from FAISS (sim≈{top_sim:.3f}).")
        else:
            # 4) Retrieval-grounded generation
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.retriever,
            )
            resp = qa.run(
                f"Intent: {json.dumps(intent)}\n"
                f"User request: {user_query}\n"
                f"Generate the minimal-cost, parameterized artifact (Ansible/CFN/Python) "
                f"or textual answer strictly grounded in the retrieved context. "
                f"If parameters are missing, assume safe low-cost defaults."
            )
            code = None
            if "```" in resp:
                try:
                    seg = resp.split("```", 2)[1]
                    code = seg.split("\n", 1)[-1].rsplit("```", 1)[0]
                except Exception:
                    code = None
            st.session_state.generated_script = code or resp

# ---------- Editor + Apply ----------
if st.session_state.generated_script:
    st.subheader("Editor")
    with st.form("editor"):
        edited = st.text_area("Review/Edit (scripts or answers)", value=st.session_state.generated_script, height=420)
        c1, c2 = st.columns(2)
        with c1:
            save = st.form_submit_button("Save/OK")
        with c2:
            run_now = st.form_submit_button("Deploy/Run")

    if save or run_now:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        artifact = classify_artifact(edited)

        if artifact == "terraform":
            folder, ext = DIR_TERRAFORM, ".tf"
        elif artifact == "cfn":
            folder, ext = DIR_CFN, ".yaml"
            edited = sanitize_cloudformation_template(edited)
        elif artifact == "ansible":
            folder, ext = DIR_ANSIBLE, ".yaml"
        elif artifact == "python":
            folder, ext = DIR_PYTHON, ".py"
        else:
            folder, ext = DIR_TEXT, ".txt"

        os.makedirs(folder, exist_ok=True)
        filename = f"artifact_{ts}{ext}"
        path = os.path.join(folder, filename)
        with open(path, "w") as f:
            f.write(edited)
        st.success(f"💾 Saved to `{path}`")

        if run_now:
            if artifact == "cfn":
                region = extract_region_from_code(edited, DEFAULT_REGION)
                logs = deploy_cloudformation(edited, f"stack{ts}", region)
            elif artifact == "ansible":
                logs = deploy_ansible_playbook(path)
            elif artifact == "python":
                logs = run_python_script(path)
            else:
                logs = ["ℹ️ This artifact type is textual/unsupported for execution."]
            st.subheader("Logs")
            for line in logs:
                st.write(line)
            st.session_state.logs.extend(logs)

# ---------- History Panel ----------
st.subheader("Chat History (newest first)")
for item in st.session_state.history:
    with st.expander(f"{item['query']}  —  intent: {item['intent']['action']} → {', '.join(item['intent']['resources'])}"):
        for i, ch in enumerate(item["chunks"], 1):
            st.code(ch)

st.caption(
    "This app always searches the FAISS vector DB first. If no high-similarity match is found, "
    "it generates grounded artifacts using retrieved context."
)
