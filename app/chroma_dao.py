"""Funzioni di accesso al database vettoriale Chroma."""

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document


class ChromaDAO:
    """Piccolo wrapper che isola la libreria Chroma dal resto del progetto."""

    def __init__(
        self,
        db_path: str | Path,
        collection_name: str,
        embedding_function: Any,
    ) -> None:
        self.client = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=str(db_path),
        )

    def find(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Cerca i chunk più simili applicando, se presenti, i filtri metadata."""
        try:
            return self.client.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filters,
            )
        except Exception:
            # Compatibilità con versioni di langchain-chroma che espongono
            # soltanto lo score di distanza grezzo.
            return self.client.similarity_search_with_score(
                query=query,
                k=k,
                filter=filters,
            )

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str],
    ) -> list[str]:
        if not documents:
            return []
        return self.client.add_documents(documents=documents, ids=ids)

    def get_all_ids(self, batch_size: int = 5000) -> list[str]:
        all_ids = []
        try:
            total = self.client._collection.count()
            for offset in range(0, total, batch_size):
                res = self.client._collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"],
                )
                ids = res.get("ids") or []
                all_ids.extend(ids)
        except Exception:
            all_ids = self.client.get(include=["metadatas"]).get("ids") or []
        return all_ids

    def get_all_documents(self, batch_size: int = 5000) -> list[Document]:
        documents = []
        try:
            total = self.client._collection.count()
            for offset in range(0, total, batch_size):
                res = self.client._collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                texts = res.get("documents") or []
                metadatas = res.get("metadatas") or []
                for index, text in enumerate(texts):
                    documents.append(
                        Document(
                            page_content=text,
                            metadata=metadatas[index] if index < len(metadatas) else {},
                        )
                    )
        except Exception:
            result = self.client.get(include=["documents", "metadatas"])
            texts = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            documents = [
                Document(
                    page_content=text,
                    metadata=metadatas[index] if index < len(metadatas) else {},
                )
                for index, text in enumerate(texts)
            ]
        return documents
