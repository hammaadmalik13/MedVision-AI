"""LLM integration and Medical RAG system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medvision.config import get_settings


@dataclass
class Citation:
    source: str
    page: int | None
    chunk_id: str
    content: str


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]


class MedicalRAGSystem:
    def __init__(self, persist_dir: str | None = None) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self._vectorstore = None
        self._embeddings = None
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

    def _init_embeddings(self):
        if self._embeddings is None:
            settings = get_settings()
            if settings.openai_api_key:
                from langchain_openai import OpenAIEmbeddings

                self._embeddings = OpenAIEmbeddings(
                    model=settings.rag_embedding_model,
                    api_key=settings.openai_api_key,
                )
            else:
                from langchain.embeddings import FakeEmbeddings

                self._embeddings = FakeEmbeddings(size=384)
        return self._embeddings

    def _init_vectorstore(self):
        if self._vectorstore is None:
            from langchain_community.vectorstores import Chroma

            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self._init_embeddings(),
            )
        return self._vectorstore

    def ingest_pdf(self, pdf_path: Path, source_name: str | None = None) -> int:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = source_name or pdf_path.name

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        vectorstore = self._init_vectorstore()
        vectorstore.add_documents(chunks)
        return len(chunks)

    def query(self, question: str, k: int = 4) -> RAGResponse:
        vectorstore = self._init_vectorstore()
        docs = vectorstore.similarity_search(question, k=k)

        citations = [
            Citation(
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                chunk_id=str(i),
                content=doc.page_content[:200],
            )
            for i, doc in enumerate(docs)
        ]

        context = "\n\n".join(f"[{c.source}] {doc.page_content}" for c, doc in zip(citations, docs, strict=True))
        answer = self._generate_answer(question, context, citations)
        return RAGResponse(answer=answer, citations=citations)

    def _generate_answer(self, question: str, context: str, citations: list[Citation]) -> str:
        settings = get_settings()
        prompt = (
            "You are a medical AI assistant. Answer based ONLY on the provided context. "
            "Include citation references like [source]. "
            "If unsure, say so. This is not medical advice.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )

        if settings.openai_api_key and settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)
            response = llm.invoke(prompt)
            return response.content

        if settings.google_api_key and settings.llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=settings.google_api_key)
            response = llm.invoke(prompt)
            return response.content

        sources = ", ".join({c.source for c in citations})
        return (
            f"Based on available medical literature [{sources}]: "
            f"The context suggests relevant information about '{question}'. "
            "Configure OPENAI_API_KEY or GOOGLE_API_KEY for full LLM responses."
        )


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.rag = MedicalRAGSystem()

    def explain_mri(self, study_metadata: dict) -> str:
        return self._call_llm(
            f"Explain this MRI study metadata in clinical terms: {study_metadata}"
        )

    def explain_segmentation(self, metrics: dict) -> str:
        return self._call_llm(
            f"Explain these brain tumor segmentation results: {metrics}"
        )

    def generate_clinical_summary(self, report_data: dict) -> str:
        return self._call_llm(
            f"Generate a clinical summary for this brain tumor report: {report_data}"
        )

    def generate_patient_report(self, report_data: dict) -> str:
        return self._call_llm(
            f"Generate a patient-friendly explanation of these results: {report_data}. "
            "Use simple language, avoid jargon."
        )

    def answer_medical_question(self, question: str, use_rag: bool = True) -> RAGResponse | str:
        if use_rag:
            return self.rag.query(question)
        return self._call_llm(question)

    def _call_llm(self, prompt: str) -> str:
        if not self.settings.allow_phi_to_llm:
            prompt = "[PHI REDACTED MODE] " + prompt[:500]

        if self.settings.openai_api_key:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=self.settings.llm_model, api_key=self.settings.openai_api_key)
            return llm.invoke(prompt).content

        if self.settings.google_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=self.settings.google_api_key)
            return llm.invoke(prompt).content

        return (
            "LLM not configured. Set OPENAI_API_KEY or GOOGLE_API_KEY in environment. "
            f"Prompt received: {prompt[:100]}..."
        )
