from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Iterable, List, Optional

from llama_cpp import Llama


def load_llm(
    model_path: Path,
    n_ctx: int = 4096,
    n_gpu_layers: Optional[int] = None,
    n_threads: Optional[int] = None,
    temperature: float = 0.2,
) -> Llama:
    if n_threads is None:
        n_threads = max(2, (os.cpu_count() or 4) // 2)
    if n_gpu_layers is None:
        if platform.system() == "Darwin":
            n_gpu_layers = 40
        else:
            n_gpu_layers = 0
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model not found at {model_path}. Run the download script in scripts/ or update --llm-path."
        )
    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        use_mmap=True,
        use_mlock=True,
        temperature=temperature,
        verbose=False,
    )


def format_prompt(question: str, passages: Iterable[str]) -> str:
    context_blocks = "\n\n".join(f"Passage {idx+1}:\n{passage}" for idx, passage in enumerate(passages))
    prompt = (
        "You are a helpful science and technology assistant. Use the passages to answer succinctly.\n\n"
        f"{context_blocks}\n\n"
        f"Question: {question}\nAnswer:"
    )
    return prompt


def generate_answer(
    llm: Llama,
    question: str,
    passages: List[str],
    max_tokens: int = 256,
) -> str:
    prompt = format_prompt(question, passages)
    response = llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=llm.temperature if hasattr(llm, "temperature") else 0.2,
        top_p=0.8,
        stop=["Question:", "\n\n"],
    )
    return response["choices"][0]["text"].strip()


__all__ = ["load_llm", "format_prompt", "generate_answer"]

