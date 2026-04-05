"""
RAG pipeline: load OTC medicine PDF, chunk, embed, retrieve with FAISS.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DEFAULT_PDF = Path(__file__).resolve().parent / "Data" / "OTC_medicines_comprehensive_list_clear.pdf"
CACHE_DIR = Path(__file__).resolve().parent / ".rag_cache"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120



def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _pdf_fingerprint(pdf_path: Path) -> str:
    st = pdf_path.stat()
    raw = f"{st.st_size}:{int(st.st_mtime)}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


class MedicineRAG:
    """Embed OTC PDF chunks and retrieve by semantic similarity."""

    def __init__(
        self,
        pdf_path: Path | None = None,
        cache_dir: Path | None = None,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        self.pdf_path = Path(pdf_path or DEFAULT_PDF)
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.embed_model_name = embed_model
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._chunks: list[str] = []
        self._dim: int = 0

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embed_model_name)
        return self._model

    def _cache_paths(self, fp: str) -> tuple[Path, Path, Path]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        base = self.cache_dir / f"idx_{fp}"
        return base.with_suffix(".faiss"), base.with_suffix(".pkl"), base.with_suffix(".meta.json")

    def _load_cache(self, fp: str) -> bool:
        fidx, fpkl, fmeta = self._cache_paths(fp)
        if not (fidx.exists() and fpkl.exists() and fmeta.exists()):
            return False
        try:
            meta = json.loads(_read_text_file(fmeta))
            if meta.get("embed_model") != self.embed_model_name:
                return False
            if meta.get("chunk_size") != CHUNK_SIZE or meta.get("chunk_overlap") != CHUNK_OVERLAP:
                return False
            self._dim = int(meta["dim"])
            self._index = faiss.read_index(str(fidx))
            with open(fpkl, "rb") as f:
                self._chunks = pickle.load(f)
            return len(self._chunks) > 0 and self._index.ntotal == len(self._chunks)
        except Exception:
            return False

    def _save_cache(self, fp: str) -> None:
        fidx, fpkl, fmeta = self._cache_paths(fp)
        if self._index is None:
            return
        faiss.write_index(self._index, str(fidx))
        with open(fpkl, "wb") as f:
            pickle.dump(self._chunks, f)
        meta = {
            "dim": self._dim,
            "embed_model": self.embed_model_name,
            "chunks": len(self._chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
        fmeta.write_text(json.dumps(meta), encoding="utf-8")

    def build_index(self, force: bool = False) -> int:
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        fp = _pdf_fingerprint(self.pdf_path)
        if not force and self._load_cache(fp):
            return len(self._chunks)

        raw = _extract_pdf_text(self.pdf_path)
        self._chunks = _chunk_text(raw)
        if not self._chunks:
            raise ValueError("No text extracted from PDF.")

        emb = self.model.encode(
            self._chunks,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self._dim = int(emb.shape[1])
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(emb.astype(np.float32))
        self._save_cache(fp)
        return len(self._chunks)

    def ensure_index(self, force: bool = False) -> int:
        if self._index is not None and self._chunks and not force:
            return len(self._chunks)
        return self.build_index(force=force)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        self.ensure_index()
        if self._index is None or not self._chunks:
            return []
        q = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        scores, idxs = self._index.search(q, min(k, len(self._chunks)))
        out: list[tuple[str, float]] = []
        for j, sc in zip(idxs[0], scores[0]):
            if 0 <= j < len(self._chunks):
                out.append((self._chunks[j], float(sc)))
        return out

    def format_context(self, query: str, k: int = 5) -> str:
        hits = self.retrieve(query, k=k)
        blocks = []
        for i, (text, score) in enumerate(hits, start=1):
            blocks.append(f"[Passage {i}] (relevance {score:.3f})\n{text}")
        return "\n\n---\n\n".join(blocks)

def build_rag_answer_prompt(
    user_input: str,
    context: str,
    history: list[dict[str, str]] | None = None,
    is_blocked: bool = False,
) -> list[dict[str, str]]:
    # If user is blocked, return only a blocked message
    if is_blocked:
        return [
            {"role": "system", "content": "You are an OTC Medicine Assistant."},
            {"role": "user", "content": f"User Input:\n{user_input}\n\nContext:\n{context}"}
        ]

    system = (
        "You are a STRICT OTC Medicine Assistant using RAG.\n\n"

"PRIMARY OBJECTIVE:\n"
"You must accurately match user symptoms with the provided context and extract ONLY relevant medicines.\n"
"You are NOT a general chatbot. You are a structured medical extractor.\n\n"

"-----------------------------------\n"
"GLOBAL RULES\n"
"-----------------------------------\n"
"- Use ONLY the provided context\n"
"- Do NOT use prior knowledge\n"
"- Do NOT guess or infer outside context\n"
"- If symptom is not clearly matched → DO NOT recommend\n"
"- Always prefer EXACT matches over partial matches\n\n"

"-----------------------------------\n"
"STEP 0: GREETING + NAME HANDLING\n"
"-----------------------------------\n"
"If user introduces name:\n"
"- Extract name\n"
"- Greet using name\n"
"- Ask ONLY for age and gender\n\n"

"-----------------------------------\n"
"STEP 1: AGE CHECK\n"
"-----------------------------------\n"
"If age < 17:\n"
"Respond ONLY:\n"
"'AGE_BLOCKED'\n\n"

"If age ≥ 17:\n"
"Proceed\n\n"

"-----------------------------------\n"
"STEP 2: PRIMARY SYMPTOM CAPTURE\n"
"-----------------------------------\n"
"Ask:\n"
"'What symptoms are you experiencing?'\n\n"

"Normalize symptom into standard form:\n"
"- fever → fever\n"
"- headache → headache\n"
"- cold → cold\n\n"

"-----------------------------------\n"
"STEP 3: FOLLOW-UP SYMPTOMS\n"
"-----------------------------------\n"
"Ask related symptoms based on primary symptom\n\n"

"-----------------------------------\n"
"STEP 4: CONTEXT MATCHING (CRITICAL)\n"
"-----------------------------------\n"
"You MUST perform these steps internally:\n\n"

"1. Scan entire context\n"
"2. Identify all medicines\n"
"3. For each medicine, extract:\n"
"   - Purpose\n"
"   - Dosage\n"
"   - Notes\n\n"

"4. MATCHING RULE:\n"
"   - Match user symptoms with 'Purpose'\n"
"   - EXACT keyword match = HIGH priority\n"
"   - Synonym match = MEDIUM priority\n"
"   - Weak/unclear match = IGNORE\n\n"

"Examples:\n"
"- fever → must match 'fever'\n"
"- headache → must match 'headache' or 'pain'\n\n"

"5. FILTER:\n"
"- Keep ONLY medicines that match symptoms\n"
"- Discard unrelated medicines\n\n"

"-----------------------------------\n"
"STEP 5: SEVERITY CHECK\n"
"-----------------------------------\n"
"If severe symptoms detected:\n"
"Respond ONLY:\n"
"'Your symptoms may indicate a serious condition. Please consult a doctor immediately.'\n\n"

"-----------------------------------\n"
"STEP 6: MEDICINE SELECTION\n"
"-----------------------------------\n"
"- Select MAXIMUM 2 BEST matched medicines\n"
"- If less than 2 available → return available ones\n"
"- If NONE → Respond:\n"
"'Based on your symptoms, it is best to consult a doctor.'\n\n"

"-----------------------------------\n"
"STEP 7: OUTPUT FORMAT\n"
"-----------------------------------\n"
"Here are suitable OTC options:\n\n"

"1. Medicine Name:\n"
"   - Purpose:\n"
"   - Dosage:\n"
"   - Additional Notes:\n\n"

"2. Medicine Name:\n"
"   - Purpose:\n"
"   - Dosage:\n"
"   - Additional Notes:\n\n"

"-----------------------------------\n"
"FINAL STEP\n"
"-----------------------------------\n"
"- Thank user using their name\n\n"

"-----------------------------------\n"
"STRICT PROHIBITIONS\n"
"-----------------------------------\n"
"- No hallucinated medicines\n"
"- No external knowledge\n"
"- No 'commonly used' suggestions\n"
"- No vague matches\n"
"- No skipping steps\n"
    )

    user = f"User Input:\n{user_input}\n\nContext:\n{context}"
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    if history:
        for msg in history:
            if msg["role"] in {"user", "assistant"}:
                messages.append(msg)

    messages.append({"role": "user", "content": user})
    return messages


def stream_openai_chat(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    from openai import OpenAI

    kw: dict[str, str] = {}
    if api_key:
        kw["api_key"] = api_key
    if base_url:
        kw["base_url"] = base_url
    client = OpenAI(**kw) if kw else OpenAI()
    return client.chat.completions.create(model=model, messages=messages, stream=True)


def answer_without_llm(query: str, hits: list[tuple[str, float]], max_snippets: int = 3) -> str:
    """Fallback when no API key: show top retrieved passages."""
    lines = [
        "No LLM API key configured. Showing the most relevant passages from the OTC list "
        "(set **GROQ_API_KEY** in `.env` and restart the app):",
        "",
    ]
    for i, (text, score) in enumerate(hits[:max_snippets], start=1):
        lines.append(f"**Passage {i}** (score {score:.3f})\n\n{text[:2000]}")
        lines.append("")
    return "\n".join(lines).strip()
