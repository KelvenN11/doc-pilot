from __future__ import annotations

from pathlib import Path


def normalize_text(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


def suggest_tags(text: str, source_file: str) -> list[str]:
    """
    Simple low-cost tag suggester.

    This is intentionally rule-based for the first version.
    Later, replace or combine this with:
    - embedding-based tag matching
    - LLM-based tag suggestion
    - human approval UI
    """

    combined = normalize_text(source_file + " " + text)

    tags: set[str] = set()

    # Course/module hints
    if "cs2040s" in combined:
        tags.add("CS2040S")
    if "cs2030s" in combined:
        tags.add("CS2030S")
    if "ma2104" in combined:
        tags.add("MA2104")
    if "es1103" in combined:
        tags.add("ES1103")

    # Graph algorithms
    if "graph" in combined:
        tags.add("Graph Algorithms")
    if "dijkstra" in combined:
        tags.add("Dijkstra")
        tags.add("Shortest Paths")
        tags.add("Graph Algorithms")
    if "bellman" in combined or "bellman-ford" in combined:
        tags.add("Bellman-Ford")
        tags.add("Shortest Paths")
        tags.add("Graph Algorithms")
    if "shortest path" in combined or "shortest paths" in combined:
        tags.add("Shortest Paths")
    if "negative edge" in combined or "negative weight" in combined:
        tags.add("Negative Edges")

    # Java / CS2030S
    if "generic" in combined or "generics" in combined:
        tags.add("Java Generics")
    if "wildcard" in combined or "? extends" in combined or "? super" in combined:
        tags.add("Java Generics")
        tags.add("Variance")
    if "stream" in combined and "java" in combined:
        tags.add("Java Streams")

    # Topology / MA2104
    if "topology" in combined:
        tags.add("Topology")
    if "hausdorff" in combined:
        tags.add("Hausdorff Spaces")
        tags.add("Topology")
    if "compact" in combined or "compactness" in combined:
        tags.add("Compactness")
        tags.add("Topology")
    if "connected" in combined or "connectedness" in combined:
        tags.add("Connectedness")
        tags.add("Topology")

    # Neural networks / AI
    if "neural network" in combined or "deep learning" in combined:
        tags.add("Neural Networks")
    if "cnn" in combined or "convolution" in combined:
        tags.add("CNN")
        tags.add("Neural Networks")
    if "receptive field" in combined:
        tags.add("Receptive Field")
        tags.add("CNN")
    if "attention" in combined or "transformer" in combined:
        tags.add("Transformers")
        tags.add("Attention")

    return sorted(tags)


def tags_to_metadata_string(tags: list[str]) -> str:
    """
    Chroma metadata values should be simple scalar values.
    Store tags as a comma-separated string for simplicity.
    """
    return ", ".join(tags)


def extract_file_level_tags(file_path: str | Path) -> list[str]:
    """
    Suggest tags from filename only.
    Useful for document-level metadata.
    """
    path = Path(file_path)
    return suggest_tags(text="", source_file=path.name)