import os
import glob
from typing import List, Tuple
import numpy as np
from langchain_together import TogetherEmbeddings, ChatTogether
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


class InMemoryVectorIndex:
	def __init__(self) -> None:
		self.embeddings: np.ndarray | None = None
		self.metadatas: List[dict] = []
		self.texts: List[str] = []

	def add(self, vectors: np.ndarray, texts: List[str], metadatas: List[dict]) -> None:
		if self.embeddings is None:
			self.embeddings = vectors.astype(np.float32)
		else:
			self.embeddings = np.vstack([self.embeddings, vectors.astype(np.float32)])
		self.texts.extend(texts)
		self.metadatas.extend(metadatas)

	def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, dict, float]]:
		if self.embeddings is None or len(self.texts) == 0:
			return []
		# cosine similarity
		A = self.embeddings
		q = query_vector.astype(np.float32)
		A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
		q_norm = q / (np.linalg.norm(q) + 1e-8)
		sims = np.dot(A_norm, q_norm)
		idx = np.argsort(-sims)[:k]
		return [(self.texts[i], self.metadatas[i], float(sims[i])) for i in idx]


class RAGPipeline:
	def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3", embed_model: str = "togethercomputer/m2-bert-80M-8k-retrieval") -> None:
		api_key = os.getenv("TOGETHER_API_KEY")
		if not api_key:
			raise RuntimeError("TOGETHER_API_KEY not set")
		model_name = os.getenv("TOGETHER_LLM_MODEL", model)
		embed_name = os.getenv("TOGETHER_EMBED_MODEL", embed_model)
		self.embedder = TogetherEmbeddings(model=embed_name)
		self.llm = ChatTogether(model=model_name, temperature=0.2)
		self.index = InMemoryVectorIndex()
		self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

	def ingest_pdfs(self, folder: str) -> int:
		pdf_paths = glob.glob(os.path.join(folder, "**/*.pdf"), recursive=True)
		all_docs: List[Document] = []
		for path in pdf_paths:
			loader = PyPDFLoader(path)
			try:
				docs = loader.load()
			except Exception:
				docs = []
			for d in docs:
				d.metadata = {"source": path, **d.metadata}
			all_docs.extend(docs)
		if not all_docs:
			return 0
		chunks = self.splitter.split_documents(all_docs)
		texts = [c.page_content for c in chunks]
		metas = [c.metadata for c in chunks]
		vectors = np.array(self.embedder.embed_documents(texts), dtype=np.float32)
		self.index.add(vectors, texts, metas)
		return len(texts)

	def retrieve(self, question: str, k: int = 5) -> List[Tuple[str, dict, float]]:
		q_vec = np.array(self.embedder.embed_query(question), dtype=np.float32)
		return self.index.search(q_vec, k=k)

	def answer(self, question: str, top_k: int = 5) -> Tuple[str, List[Tuple[str, dict, float]]]:
		retrieved = self.retrieve(question, k=top_k)
		context_blocks = []
		for text, meta, score in retrieved:
			src = meta.get("source", "")
			context_blocks.append(f"[Source: {src} | score {score:.3f}]\n{text}")
		context = "\n\n".join(context_blocks) or ""
		prompt = (
			"You are a helpful assistant. Answer the question using the provided context snippets.\n"
			"If the answer isn't in the context, say you don't know and suggest where to look.\n\n"
			f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
		)
		resp = self.llm.invoke(prompt)
		return (resp.content if hasattr(resp, "content") else str(resp), retrieved)


rag_pipeline: RAGPipeline | None = None


def get_rag() -> RAGPipeline:
	global rag_pipeline
	if rag_pipeline is None:
		rag_pipeline = RAGPipeline()
	return rag_pipeline