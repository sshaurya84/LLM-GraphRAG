"""
Utilities for downloading a focused science & technology corpus from Wikipedia.

The ingestion pipeline:
1. Fetch a fixed list of seed pages.
2. Collect up to N first-hop links that match science/technology keywords.
3. Persist each page as JSON containing metadata, raw wikitext, and cleaned text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set

import mwparserfromhell
import wikipediaapi

logger = logging.getLogger(__name__)


#: Primary domain seeds provided in the project plan.
DEFAULT_SEEDS: List[str] = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Reinforcement learning",
    "Large language model",
    "Natural language processing",
    "Computer vision",
    "Robotics",
    "Autonomous robot",
    "Transformer (machine learning)",
    "Knowledge graph",
    "Graph neural network",
    "Graph database",
    "Information retrieval",
    "Recommender system",
    "Data mining",
    "Distributed computing",
    "High-performance computing",
    "Parallel computing",
    "Edge computing",
    "Cloud computing",
    "Federated learning",
    "Explainable artificial intelligence",
    "Computer security",
    "Cyber-physical system",
    "Internet of things",
    "Human–computer interaction",
    "Computer graphics",
    "Augmented reality",
    "Virtual reality",
    "Operating system",
    "Compiler",
    "Semiconductor device",
    "Microprocessor",
    "Quantum computing",
    "Supercomputer",
    "Computational biology",
    "Bioinformatics",
    "Computational chemistry",
]

# Keywords to retain during first-hop expansion.
SCIENCE_TECH_KEYWORDS = {
    "computer",
    "computing",
    "algorithm",
    "engineering",
    "science",
    "technology",
    "physics",
    "mathematics",
    "graph",
    "quantum",
    "semiconductor",
    "chip",
    "model",
    "learning",
    "artificial intelligence",
    "software",
    "hardware",
    "data",
    "information",
    "cloud",
    "edge",
    "security",
    "cyber",
    "robot",
    "automation",
    "sensor",
    "computational",
    "system",
    "parallel",
    "neural",
    "network",
    "hci",
    "augmented",
    "virtual",
    "bio",
}

# Page titles that are allowable even if they do not match keyword filters.
ALLOWLIST_TITLES = {
    "OpenAI",
    "Anthropic",
    "DeepMind",
    "Nvidia",
    "Google",
    "Microsoft",
    "IBM",
    "Intel",
    "Meta Platforms",
    "Amazon (company)",
    "Apple Inc.",
    "Stanford University",
    "Massachusetts Institute of Technology",
    "Carnegie Mellon University",
    "University of California, Berkeley",
    "University of Oxford",
    "University of Cambridge",
    "National Aeronautics and Space Administration",
}


@dataclass
class PageRecord:
    """Structured representation of a Wikipedia page."""

    title: str
    url: str
    summary: str
    clean_text: str
    raw_wikitext: str
    outlinks: List[str]
    categories: List[str]
    source_seed: str
    depth: int


def _build_wiki_client(language: str = "en") -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(
        user_agent="GraphRAG-Metal/0.1 (https://github.com/your-org/GraphRAG)",
        language=language,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def _clean_text(wikitext: str) -> str:
    """Remove markup and tables using mwparserfromhell."""
    parsed = mwparserfromhell.parse(wikitext)
    # Drop infoboxes, tables, and references
    for template in parsed.filter_templates():
        if template.name.lower().strip().startswith(("infobox", "cite", "navbox")):
            parsed.remove(template)
    for table in parsed.filter_tags(matches=lambda node: str(node.tag).lower() == "table"):
        parsed.remove(table)
    # Remove reference tags
    for tag in parsed.filter_tags(matches=lambda node: str(node.tag).lower() in {"ref", "math"}):
        parsed.remove(tag)
    return parsed.strip_code().strip()


def _is_science_or_tech(category_titles: Iterable[str]) -> bool:
    for cat in category_titles:
        normalized = cat.lower()
        if any(keyword in normalized for keyword in SCIENCE_TECH_KEYWORDS):
            return True
    return False


def _filter_first_hop_links(
    titles: Iterable[str],
    allowlist: Set[str],
    max_links: int,
) -> List[str]:
    filtered: List[str] = []
    for title in titles:
        norm_title = title.strip()
        if not norm_title or norm_title.startswith("Help:") or norm_title.startswith("File:"):
            continue
        if norm_title in allowlist:
            filtered.append(norm_title)
            continue
        lower_title = norm_title.lower()
        if any(keyword in lower_title for keyword in SCIENCE_TECH_KEYWORDS):
            filtered.append(norm_title)
        if len(filtered) >= max_links:
            break
    return filtered


def download_corpus(
    output_path: Path,
    seeds: Optional[Iterable[str]] = None,
    first_hop_limit: int = 20,
    second_hop_limit: int = 8,
) -> List[PageRecord]:
    """
    Fetch the seed corpus and persist records to ``output_path`` (JSONL file).

    Returns a list of ``PageRecord`` items for downstream processing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wiki = _build_wiki_client()
    seeds = list(seeds or DEFAULT_SEEDS)
    seen_titles: Set[str] = set()
    records: List[PageRecord] = []

    def _process_page(title: str, depth: int, source_seed: str) -> Optional[PageRecord]:
        if title in seen_titles:
            logger.debug("Skipping already-seen page %s", title)
            return None
        page = wiki.page(title)
        if not page.exists():
            logger.warning("Page does not exist: %s", title)
            return None
        categories = list(page.categories.keys())
        if depth > 0 and not _is_science_or_tech(categories):
            if title not in ALLOWLIST_TITLES:
                logger.debug("Skipping non-science/tech page %s", title)
                return None
        clean_text = _clean_text(page.text)
        record = PageRecord(
            title=page.title,
            url=page.fullurl,
            summary=page.summary,
            clean_text=clean_text,
            raw_wikitext=page.text,
            outlinks=list(page.links.keys()),
            categories=categories,
            source_seed=source_seed,
            depth=depth,
        )
        seen_titles.add(page.title)
        records.append(record)
        return record

    for seed in seeds:
        logger.info("Processing seed page: %s", seed)
        seed_record = _process_page(seed, depth=0, source_seed=seed)
        if not seed_record:
            continue

        first_hop_titles = _filter_first_hop_links(
            seed_record.outlinks,
            allowlist=ALLOWLIST_TITLES | set(seeds),
            max_links=first_hop_limit,
        )
        for title in first_hop_titles:
            first_hop_record = _process_page(title, depth=1, source_seed=seed)
            if not first_hop_record or second_hop_limit <= 0:
                continue
            second_hop_titles = _filter_first_hop_links(
                first_hop_record.outlinks,
                allowlist=ALLOWLIST_TITLES | set(seeds),
                max_links=second_hop_limit,
            )
            for second_title in second_hop_titles:
                _process_page(second_title, depth=2, source_seed=seed)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    logger.info("Saved %d pages to %s", len(records), output_path)
    return records


def load_corpus(path: Path) -> List[PageRecord]:
    """Convenience loader for previously saved records."""
    records: List[PageRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            records.append(PageRecord(**payload))
    return records


__all__ = [
    "ALLOWLIST_TITLES",
    "DEFAULT_SEEDS",
    "PageRecord",
    "download_corpus",
    "load_corpus",
]

