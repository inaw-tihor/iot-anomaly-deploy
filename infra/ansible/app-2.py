# ============================
# RAG Cloud Agent – FAISS Edition (Single File)
# ============================
# - Constants consolidated at the top
# - FAISS local vector store (save/load)
# - Preserves prior features (CFN/Ansible/Python generation & execution)
# - Adds Shell/CLI command/script execution (with safety toggle)
# - Exact-match reuse via FAISS similarity
# - Editor + logs + chat history + related chunks
# - MkDocs site loader (via Material search index)

def inject_print_css():
    st.markdown(
        """
        <style>
        /* Print-only rules */
        @media print {
          /* Hide Streamlit chrome & interactive controls */
          [data-testid="stSidebar"],
          [data-testid="stToolbar"],
          [data-testid="stHeader"],
          footer,
          .stButton, .stDownloadButton, button,
          textarea, input, select,
          .stTextInput, .stTextArea, .stNumberInput,
          .stSlider, .stDateInput, .stFileUploader, .stMultiSelect, .stRadio, .stSelectbox {
            display: none !important;
          }

          /* Use full width, add margins for paper */
          [data-testid="stAppViewContainer"] { margin: 0 !important; padding: 0 !important; }
          .block-container { padding: 12mm !important; }

          /* Stack columns to avoid horizontal compression overlap */
          [data-testid="column"] { display: block !important; width: 100% !important; }

          /* Expanders: print content only; keep it open */
          [data-testid="stExpander"] summary { display: none !important; }
          [data-testid="stExpander"] div[role="region"] { display: block !important; }

          /* Avoid overlaps & clipping */
          [data-testid="stAppViewContainer"],
          .block-container,
          [data-testid="stVerticalBlock"],
          [data-testid="stMarkdownContainer"] {
            overflow: visible !important;
          }

          /* Page-break hygiene */
          [data-testid="stVerticalBlock"],
          [data-testid="stMarkdownContainer"],
          [data-testid="stCodeBlock"] {
            break-inside: avoid-page !important;
            page-break-inside: avoid !important;
          }
          h1, h2, h3 { page-break-after: avoid !important; }
          pre, code { white-space: pre-wrap !important; word-break: break-word; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- CONSTANTS ----------
APP_TITLE = "CookBook Agent"
DEFAULT_REGION = "us-east-1"

# Embeddings / Vector DB
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
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
DIR_PYTHON = "scripts/others"   # per your requirement
DIR_SHELL = "scripts/shell"     # NEW: for CLI scripts
DIR_TEXT = "scripts/other"

# CLI execution safety
ALLOW_CLI_DEFAULT = False       # NEW: default off

# ---------- IMPORTS ----------
import sys
import types
import os
import stat
import tempfile
import git
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import shlex

import boto3
from botocore.exceptions import ClientError

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, PyPDFium2Loader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import requests
from urllib.parse import urljoin

# ---------- Torch Fix (some HF envs need this) ----------
torch_classes_dummy = types.ModuleType("torch.classes")
torch_classes_dummy.__path__ = []
sys.modules["torch.classes"] = torch_classes_dummy

# ---------- UI / SESSION ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")  # if you use this, keep it first
st.title(APP_TITLE)
# inject print styles once
if "_print_css" not in st.session_state:
    inject_print_css()
    st.session_state["_print_css"] = True

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
allow_cli = st.sidebar.checkbox("Allow CLI execution (dangerous)", value=ALLOW_CLI_DEFAULT)  # NEW

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
        items = j.get("docs") or j.get("index", {}).get("docs") or []
        urls = []
        seen = set()
        for it in items:
            loc = it.get("location", "")
            page_url = urljoin(base_url, loc.split("#", 1)[0])
            if page_url not in seen and page_url.startswith(base_url):
                urls.append(page_url)
                seen.add(page_url)
        if not urls:
            urls = [base_url]
        loader = WebBaseLoader(urls)
        docs = loader.load()
    except Exception:
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


import shlex

def _inject_ansible_python(cmd_str: str, py_path: str) -> str:
    """
    Ensure ansible-playbook runs with our interpreter by:
    1) Prepending ANSIBLE_PYTHON_INTERPRETER=...
    2) Appending -e ansible_python_interpreter=...
    (Only if user hasn't already set them.)
    """
    has_env = re.search(r'\bANSIBLE_PYTHON_INTERPRETER\b', cmd_str)
    has_extra = re.search(r'\bansible_python_interpreter\s*=', cmd_str)
    if not has_env:
        cmd_str = f'ANSIBLE_PYTHON_INTERPRETER={shlex.quote(py_path)} ' + cmd_str
    if not has_extra:
        cmd_str = cmd_str + f' -e ansible_python_interpreter={shlex.quote(py_path)}'
    return cmd_str

def _normalize_aws_env_for_ansible(env: dict, logs: list) -> dict:
    """
    Resolve 'profile + access keys' conflicts for ansible-playbook.
    Prefers AWS profile if both are present. Also sets region defaults and config load.
    """
    profile = env.get("AWS_PROFILE") or env.get("AWS_DEFAULT_PROFILE")
    key_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"]
    has_keys = any(env.get(k) for k in key_vars)

    # If both are present, choose profile and drop keys to avoid the boto3 error.
    if profile and has_keys:
        for k in key_vars:
            if env.pop(k, None) is not None:
                pass
        logs.append(f"ℹ️ Detected AWS profile '{profile}' and env keys; removed env keys to avoid conflict.")

    # Ensure region is present
    if not env.get("AWS_REGION") and not env.get("AWS_DEFAULT_REGION"):
        env["AWS_DEFAULT_REGION"] = DEFAULT_REGION
        logs.append(f"ℹ️ Set AWS_DEFAULT_REGION={DEFAULT_REGION}")

    # Tell the SDK to load ~/.aws/config for profiles/regions
    env["AWS_SDK_LOAD_CONFIG"] = "1"
    return env

def run_shell(path: str = None, command: str = None) -> list[str]:
    """
    Execute a shell script file or a one-liner command in /bin/bash -lc.
    For ansible-playbook, forcibly inject our Python and normalize AWS env to avoid
    'Passing both a profile and access tokens is not supported'.
    """
    logs = []
    env = os.environ.copy()

    # quick self-check: does this Python have boto3/botocore?
    try:
        import importlib
        importlib.import_module("boto3")
        importlib.import_module("botocore")
    except Exception as e:
        logs.append(f"⚠️ Current Python ({sys.executable}) may miss boto3/botocore: {e}")

    try:
        if command:
            base_cmd = command.strip()
            if re.match(r'^(ansible-playbook)(\s|$)', base_cmd):
                base_cmd = _inject_ansible_python(base_cmd, sys.executable)
                env = _normalize_aws_env_for_ansible(env, logs)
            logs.append(f"🖥️ Running command:\n{base_cmd}")
            result = subprocess.run(
                ["/bin/bash", "-lc", base_cmd],
                capture_output=True,
                text=True,
                env=env,
            )

        elif path:
            logs.append(f"🖥️ Executing script: {path}")
            # sniff first 4KB to see if it calls ansible-playbook
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(4096)
                if re.search(r'(^|\s)ansible-playbook(\s|$)', head):
                    env = _normalize_aws_env_for_ansible(env, logs)
                    wrapped = f'ANSIBLE_PYTHON_INTERPRETER={shlex.quote(sys.executable)} "{path}" -e ansible_python_interpreter={shlex.quote(sys.executable)}'
                    logs.append(f"ℹ️ Rewritten for Ansible interpreter:\n{wrapped}")
                    cmd_to_run = wrapped
                else:
                    cmd_to_run = f'"{path}"'
            except Exception:
                cmd_to_run = f'"{path}"'
            try:
                os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
            except Exception:
                pass
            result = subprocess.run(
                ["/bin/bash", "-lc", cmd_to_run],
                capture_output=True,
                text=True,
                env=env,
            )
        else:
            return ["⚠️ Nothing to run."]

        logs.append("🔧 STDOUT:\n" + (result.stdout or ""))
        if result.stderr:
            logs.append("⚠️ STDERR:\n" + result.stderr)
        logs.append("✅ Completed successfully." if result.returncode == 0 else f"❌ Exit {result.returncode}")
    except FileNotFoundError:
        logs.append("❌ '/bin/bash' not found. Ensure a POSIX shell is available.")
    except Exception as e:
        logs.append(f"❌ Shell run error: {str(e)}")
    return logs


def classify_artifact(text: str) -> str:
    """
    Classify editor content into known artifact types.
    Prioritize explicit CLI commands (e.g., 'ansible-playbook') over generic 'ansible' keyword.
    """
    t = text.strip()
    tl = t.lower()

    # 1) Shell / CLI (explicit commands or shebang) — check FIRST
    if re.match(r'^(ansible-playbook|aws|kubectl|helm|bash|sh|terraform|git|make|pip|python3?|docker|gh)\b', tl):
        return "shell"
    if any(marker in tl for marker in ["#!/bin/bash", "#!/usr/bin/env bash", "#!/bin/sh"]):
        return "shell"

    # 2) CloudFormation
    if "awstemplateformatversion" in tl or "aws::" in tl:
        return "cfn"

    # 3) Ansible YAML (structure-based, not just the word "ansible")
    if re.search(r'(?m)^(---\s*)?(\s*-\s*)?hosts:\s*', t) or re.search(r'(?m)^\s*tasks:\s*', t):
        return "ansible"

    # 4) Terraform
    if re.search(r'(?m)^\s*terraform\s*{', t) or re.search(r'(?m)^\s*resource\s+"', t):
        return "terraform"

    # 5) Python / boto3
    if re.search(r'(?m)^\s*import\s+boto3\b', t) or "boto3." in t:
        return "python"

    # Fallback
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
    elif any(c in q for c in ["cli", "shell", "bash", "sh", "command"]):
        artifact_hint = "shell"

    return {"action": action, "resources": resources or ["generic"], "artifact_hint": artifact_hint}

RESOURCE_ORDER = ["vpc", "subnet", "sg", "iam", "s3", "rds", "ec2"]

def plan_steps(intent: Dict[str, Any]) -> List[str]:
    if intent["action"] != "create":
        return intent["resources"]
    wanted = [r for r in RESOURCE_ORDER if r in intent["resources"]] or intent["resources"]
    return wanted

def load_pdf_robust(path: str):
    """Try PDFium first (most robust), then fall back to PyPDF."""
    try:
        return PyPDFium2Loader(path).load()
    except Exception as e1:
        st.warning(f"PDFium parse failed: {e1}. Falling back to PyPDF…")
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
                    # If base of an MkDocs site, crawl all pages; else load the URL as-is
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
    model_kwargs={"device": EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

def build_faiss_index(docs: List[Document], save_dir: str) -> FAISS:
    vs = FAISS.from_documents(docs, embeddings)
    os.makedirs(save_dir, exist_ok=True)
    vs.save_local(save_dir)
    return vs

def load_faiss_index(load_dir: str) -> FAISS:
    if not os.path.isdir(load_dir):
        raise FileNotFoundError(f"Vector DB index directory not found: {load_dir}")
    # allow_dangerous_deserialization=True is required due to pickled docstore
    return FAISS.load_local(load_dir, embeddings, allow_dangerous_deserialization=True)

col_build, col_load = st.columns(2)
with col_build:
    if st.button("Build Vector Index"):
        if st.session_state.chunks:
            st.session_state.vectorstore = build_faiss_index(st.session_state.chunks, FAISS_DIR)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
            st.success(f"✅ Built & saved Vector index → {FAISS_DIR}")
        else:
            st.warning("No chunks to embed.")

with col_load:
    if st.button("Load Vector Index"):
        try:
            st.session_state.vectorstore = load_faiss_index(FAISS_DIR)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
            st.success(f"✅ Loaded Vector index from {FAISS_DIR}")
        except Exception as e:
            st.error(f"❌ Error loading Vector: {e}")

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
            st.success(f"✅ Reused high-similarity match from Vector DB (sim≈{top_sim:.3f}).")
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
                f"Generate the minimal-cost, parameterized artifact (Ansible/CFN/Python/CLI) "
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
        elif artifact == "shell":
            folder, ext = DIR_SHELL, ".sh"
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
            elif artifact == "shell":
                if allow_cli:
                    is_one_liner = "\n" not in edited.strip() and not edited.strip().startswith("#!")
                    logs = run_shell(command=edited.strip()) if is_one_liner else run_shell(path=path)
                else:
                    logs = ["🚫 CLI execution is disabled. Enable 'Allow CLI execution (dangerous)' in the sidebar."]
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
    "This app always searches the vector DB first. If no high-similarity match is found, "
    "it generates grounded artifacts using retrieved context. CLI execution is gated by a safety toggle."
)
