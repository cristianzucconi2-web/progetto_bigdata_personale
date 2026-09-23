"""Creazione del modello usato per calcolare gli embedding."""

import torch
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings(local_files_only: bool = False) -> HuggingFaceEmbeddings:
    """Carica il modello di embedding su GPU, quando disponibile, oppure su CPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "local_files_only": local_files_only,
        },
        encode_kwargs={"normalize_embeddings": True},
    )
