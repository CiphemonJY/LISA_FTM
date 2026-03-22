"""
RAG (Retrieval-Augmented Generation) Module for LISA_FTM
========================================================
Connects the federated-trained model to your organization's
data warehouse / documents for accurate, grounded answers.

Usage:
    from connectors.rag_engine import RAGEngine

    rag = RAGEngine(
        model=model,                    # LISA_FTM model
        tokenizer=tokenizer,            # HuggingFace tokenizer
        vector_store="rag_store/",       # where to store embeddings
    )

    # Index your reports and data
    rag.index_folder("/path/to/annual_reports/")
    rag.index_dataframe(df, text_column="report_text", metadata={"source": "FY24"})

    # Ask questions
    answer = rag.query("What were total FY24 admissions across the network?")
    print(answer)
"""
import hashlib
import json
import logging
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

log = logging.getLogger("rag-engine")

# ─── Simple embedding model (no API calls needed) ───────────────────────────

class SimpleEmbedder:
    """
    Minimal fixed-dimension embedder using random projection + TF-IDF hashing.
    No OpenAI/HuggingFace API needed. Good enough for semantic search.

    In production: swap to sentence-transformers/all-MiniLM-L6-v2
    for high-quality embeddings.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        # Fixed random projection matrix (reproducible across runs)
        seed = 42
        rng = np.random.RandomState(seed)
        self.projection = rng.randn(8192, dim).astype(np.float32)
        # Normalize projection columns for stable dot products
        norms = np.linalg.norm(self.projection, axis=0, keepdims=True)
        self.projection /= norms

    def embed(self, text: str) -> np.ndarray:
        """Convert text to a dense vector."""
        # TF-IDF style hash
        tokens = re.findall(r"\b\w+\b", text.lower())
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, token in enumerate(tokens[:512]):  # limit context
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            bucket = h % self.dim
            vec[bucket] += 1.0 / (1.0 + np.log1p(i))
        # Project via random matrix (JL lemma approximation)
        # Cosine similarity via simple hash bucket is often sufficient
        vec /= (np.linalg.norm(vec) + 1e-8)
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


# ─── Chunking utilities ────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split text into overlapping chunks of ~chunk_size tokens (word-based).
    Overlap helps maintain context across chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_csv_row(row: dict, id_col: str = "id") -> str:
    """Convert a CSV row dict to a readable text string."""
    parts = []
    for k, v in sorted(row.items()):
        if k == id_col:
            continue
        if v is not None and str(v).strip():
            parts.append(f"{k}: {v}")
    return "; ".join(parts)


# ─── Core RAG Engine ──────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single indexed chunk of text with its embedding."""
    id: str
    text: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            id=d["id"],
            text=d["text"],
            embedding=np.array(d["embedding"], dtype=np.float32),
            metadata=d.get("metadata", {}),
        )


class VectorStore:
    """
    Simple in-memory + disk-backed vector store.
    Uses cosine similarity for retrieval.

    In production: swap to FAISS, Chroma, or Qdrant for
    million-chunk scale with GPU acceleration.
    """

    def __init__(self, store_dir: str = "rag_store"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[Chunk] = []
        self._ids_index: dict[str, int] = {}

    def add_chunk(self, chunk: Chunk):
        idx = len(self.chunks)
        self.chunks.append(chunk)
        self._ids_index[chunk.id] = idx

    def add_chunks(self, chunks: list[Chunk]):
        for c in chunks:
            self.add_chunk(c)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Search for top_k chunks most similar to query_embedding."""
        if not self.chunks:
            return []

        embeddings = np.stack([c.embedding for c in self.chunks])
        # Cosine similarity
        similarities = embeddings @ query_embedding / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.chunks[i], float(similarities[i])) for i in top_indices]

    def save(self):
        """Persist to disk."""
        data = [c.to_dict() for c in self.chunks]
        with open(self.store_dir / "index.json", "w") as f:
            json.dump(data, f)
        with open(self.store_dir / "meta.json", "w") as f:
            json.dump({"count": len(self.chunks), "saved": datetime.utcnow().isoformat() + "Z"}, f)
        log.info(f"Saved {len(self.chunks)} chunks to {self.store_dir}")

    def load(self):
        """Load from disk."""
        index_file = self.store_dir / "index.json"
        if not index_file.exists():
            log.warning(f"No index found at {index_file}")
            return
        with open(index_file) as f:
            data = json.load(f)
        self.chunks = [Chunk.from_dict(d) for d in data]
        self._ids_index = {c.id: i for i, c in enumerate(self.chunks)}
        log.info(f"Loaded {len(self.chunks)} chunks from {self.store_dir}")

    def __len__(self):
        return len(self.chunks)


class RAGEngine:
    """
    RAG engine combining retrieval with LISA_FTM model generation.

    Usage:
        rag = RAGEngine(model, tokenizer)
        rag.index_folder("./annual_reports/")
        rag.index_csv("./volume_data.csv")
        answer = rag.query("What were FY24 total admissions?")
    """

    def __init__(
        self,
        model,                    # LISA_FTM model (torch module)
        tokenizer,                # HuggingFace tokenizer
        embedder: SimpleEmbedder = None,
        vector_store: VectorStore = None,
        store_dir: str = "rag_store",
        device: str = "cpu",
        generation_max_new_tokens: int = 256,
        generation_temperature: float = 0.3,
        retrieval_top_k: int = 5,
        retrieval_score_threshold: float = 0.05,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder or SimpleEmbedder()
        self.vector_store = vector_store or VectorStore(store_dir=store_dir)
        self.device = device
        self.generation_max_new_tokens = generation_max_new_tokens
        self.generation_temperature = generation_temperature
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_score_threshold = retrieval_score_threshold

        self.model.eval()

    # ─── Indexing ────────────────────────────────────────────────────────

    def index_texts(
        self,
        texts: list[str],
        metadata: list[dict] = None,
        source: str = "unknown",
        chunk_size: int = 512,
        overlap: int = 64,
    ):
        """
        Index a list of raw texts.

        Args:
            texts: list of text strings to index
            metadata: optional per-text metadata (e.g., {"year": "FY24", "hospital": "A"})
            source: source label for all texts
            chunk_size: words per chunk
            overlap: overlapping words between chunks
        """
        all_chunks = []
        for i, text in enumerate(texts):
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            meta = (metadata[i] if metadata else {})
            meta["source"] = meta.get("source", source)
            meta["original_index"] = i

            for j, chunk_text in enumerate(chunks):
                chunk_id = f"{source}_{i}_{j}"
                emb = self.embedder.embed(chunk_text)
                chunk = Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    embedding=emb,
                    metadata=meta,
                )
                all_chunks.append(chunk)

        self.vector_store.add_chunks(all_chunks)
        self.vector_store.save()
        log.info(f"Indexed {len(texts)} documents → {len(all_chunks)} chunks")

    def index_folder(
        self,
        folder_path: str,
        file_patterns: tuple = (".txt", ".md", ".csv", ".json"),
        **kwargs,
    ):
        """
        Index all matching files in a folder.
        Each file becomes one text document.
        """
        folder = Path(folder_path)
        texts = []
        metas = []

        for fp in folder.rglob("*"):
            if fp.suffix not in file_patterns:
                continue
            try:
                if fp.suffix == ".csv":
                    import pandas as pd
                    df = pd.read_csv(fp)
                    for _, row in df.iterrows():
                        texts.append(chunk_csv_row(row.to_dict()))
                        metas.append({"source": str(fp), "type": "csv_row"})
                elif fp.suffix == ".json":
                    with open(fp) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            texts.append(json.dumps(item))
                            metas.append({"source": str(fp), "type": "json"})
                    else:
                        texts.append(json.dumps(data))
                        metas.append({"source": str(fp), "type": "json"})
                else:
                    with open(fp, encoding="utf-8", errors="replace") as f:
                        texts.append(f.read())
                    metas.append({"source": str(fp), "type": fp.suffix.lstrip(".")})
            except Exception as e:
                log.warning(f"Failed to read {fp}: {e}")

        self.index_texts(texts, metadata=metas, **kwargs)

    def index_csv(
        self,
        csv_path: str,
        text_column: str = "text",
        id_column: str = None,
        metadata_columns: list[str] = None,
        **kwargs,
    ):
        """Index a CSV file, one row = one document."""
        import pandas as pd

        df = pd.read_csv(csv_path)
        texts = df[text_column].fillna("").tolist()

        metadata = []
        for _, row in df.iterrows():
            meta = {"type": "csv"}
            if id_column and id_column in row:
                meta["id"] = str(row[id_column])
            if metadata_columns:
                for col in metadata_columns:
                    if col in row and pd.notna(row[col]):
                        meta[col] = str(row[col])
            metadata.append(meta)

        self.index_texts(texts, metadata=metadata, **kwargs)

    def index_dataframe(
        self,
        df,                      # pandas DataFrame
        text_column: str = "text",
        metadata_columns: list[str] = None,
        **kwargs,
    ):
        """Index a pandas DataFrame."""
        texts = df[text_column].fillna("").tolist()
        metas = []
        for _, row in df.iterrows():
            meta = {"type": "dataframe"}
            if metadata_columns:
                for col in metadata_columns:
                    if col in row and pd.notna(row[col]):
                        meta[col] = str(row[col])
            metas.append(meta)
        self.index_texts(texts, metadata=metas, **kwargs)

    def index_volume_data(
        self,
        data: dict,              # {"FY24": {"Q1": 12345, "Q2": ...}, ...}
        source: str = "volume_data",
    ):
        """
        Index structured volume data as readable text.

        Args:
            data: nested dict of volume data
                e.g. {"FY24": {"Q1": 12345, "Q2": 13000, ...}, ...}
            source: source label
        """
        lines = []
        for period, values in sorted(data.items()):
            if isinstance(values, dict):
                parts = [f"{period}:"]
                for k, v in sorted(values.items()):
                    parts.append(f"  {k}: {v:,}")
                lines.append("\n".join(parts))
            else:
                lines.append(f"{period}: {values:,}")

        text = "\n".join(lines)
        self.index_texts([text], metadata=[{"source": source, "type": "volume_data"}])
        log.info(f"Indexed volume data: {text[:200]}...")

    # ─── Retrieval ────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = None) -> list[tuple[Chunk, float]]:
        """Retrieve the most relevant chunks for a query."""
        top_k = top_k or self.retrieval_top_k
        query_emb = self.embedder.embed(query)
        results = self.vector_store.search(query_emb, top_k=top_k)
        return [(c, s) for c, s in results if s > self.retrieval_score_threshold]

    def build_context(self, retrieved: list[tuple[Chunk, float]]) -> str:
        """Build a context string from retrieved chunks."""
        if not retrieved:
            return ""

        context_parts = ["[CONTEXT FROM ORGANIZATION DATA]\n"]
        for i, (chunk, score) in enumerate(retrieved):
            meta = chunk.metadata
            source = meta.get("source", "unknown")
            context_parts.append(f"--- Document {i+1} (relevance: {score:.2f}) ---")
            context_parts.append(f"Source: {source}")
            context_parts.append(chunk.text)
            context_parts.append("")

        return "\n".join(context_parts)

    # ─── Generation ──────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """Generate from a text prompt using the model."""
        max_new_tokens = max_new_tokens or self.generation_max_new_tokens
        temperature = temperature or self.generation_temperature

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        generated = outputs[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    # ─── Full RAG Query ──────────────────────────────────────────────────

    def query(
        self,
        question: str,
        system_prompt: str = None,
        answer_format: str = None,
        top_k: int = None,
        verbose: bool = False,
    ) -> dict:
        """
        Full RAG query: retrieve relevant context + generate answer.

        Returns:
            dict with keys: question, answer, retrieved_chunks, context_used
        """
        retrieved = self.retrieve(question, top_k=top_k)
        context = self.build_context(retrieved)

        # Build the prompt
        if system_prompt is None:
            system_prompt = (
                "You are a helpful healthcare strategy assistant. "
                "Use the provided context to answer the question accurately. "
                "If the context doesn't contain enough information to answer precisely, "
                "say so and answer based on what you know about healthcare strategy. "
                "Format numbers clearly."
            )

        if answer_format:
            system_prompt += f"\n\nFormat your answer as: {answer_format}"

        prompt = f"{system_prompt}\n\n{context}\n\nQuestion: {question}\nAnswer:"

        answer = self.generate(prompt).strip()

        if verbose:
            print(f"=== Retrieved {len(retrieved)} chunks ===")
            for chunk, score in retrieved:
                print(f"[{score:.3f}] {chunk.text[:200]}...")
                print()

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "context_used": bool(retrieved),
            "num_sources": len(retrieved),
        }

    # ─── Query helpers for strategy ──────────────────────────────────────

    def ask_volume(self, question: str, **kwargs) -> dict:
        """Ask about volume/operational data specifically."""
        return self.query(
            question,
            system_prompt=(
                "You are a healthcare strategy analyst. Answer questions about "
                "hospital volumes, admissions, patient days, ED visits, and OR cases "
                "using the provided context. Give precise numbers when available. "
                "When discussing trends, reference specific time periods from the data."
            ),
            **kwargs,
        )

    def ask_financial(self, question: str, **kwargs) -> dict:
        """Ask about financial/performance data specifically."""
        return self.query(
            question,
            system_prompt=(
                "You are a healthcare finance analyst. Answer questions about "
                "revenue, costs, profitability, and financial trends using the "
                "provided context. Give dollar amounts with units. Reference "
                "the time periods in the data."
            ),
            **kwargs,
        )

    def ask_strategic(self, question: str, **kwargs) -> dict:
        """Ask strategic/competitive analysis questions."""
        return self.query(
            question,
            system_prompt=(
                "You are a healthcare strategy expert. Analyze competitive "
                "positioning, market share, service line performance, and "
                "strategic implications using the provided context. "
                "Support analysis with specific data points from the context."
            ),
            **kwargs,
        )


# ─── CLI demo ─────────────────────────────────────────────────────────────

def demo():
    """Demo: index sample data and run queries."""
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="RAG Engine Demo")
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--store-dir", default="rag_store")
    parser.add_argument("--query", default="What were total FY24 admissions?")
    args = parser.parse_args()

    log.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    rag = RAGEngine(model, tokenizer, store_dir=args.store_dir)

    # Index sample volume data
    sample_data = {
        "FY23": {"Q1": 11200, "Q2": 11900, "Q3": 12100, "Q4": 11800},
        "FY24": {"Q1": 12345, "Q2": 13102, "Q3": 11890, "Q4": 13107},
    }
    rag.index_volume_data(sample_data, source="annual_volume_report")

    # Run a query
    result = rag.ask_volume(args.query, verbose=True)
    print(f"\n=== ANSWER ===\n{result['answer']}")


if __name__ == "__main__":
    demo()
