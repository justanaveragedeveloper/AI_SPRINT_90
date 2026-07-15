import pytest

from day13_generator import RAGContextAssembler


def test_nominal_prompt_generation():
    assembler = RAGContextAssembler(max_chars=2000)

    query = "What is the scale factor of SQ8 quantization?"

    chunks = [
        "SQ8 quantization compresses float32 dimensions " "down to exactly 1 byte."
    ]

    prompt = assembler.assemble_prompt(
        query=query,
        context_chunks=chunks,
    )

    assert "SYSTEM INSTRUCTIONS:" in prompt
    assert "INSUFFICIENT_LOCAL_CONTEXT" in prompt
    assert query in prompt
    assert "SQ8 quantization compresses float32" in prompt


def test_constructor_validation():
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        RAGContextAssembler(0)


def test_empty_query_rejection():
    assembler = RAGContextAssembler()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        assembler.assemble_prompt(
            "   ",
            ["valid context"],
        )


def test_query_budget_validation():
    assembler = RAGContextAssembler(max_chars=100)

    oversized_query = "A" * 150

    with pytest.raises(
        ValueError,
        match="too small",
    ):
        assembler.assemble_prompt(
            oversized_query,
            [],
        )


def test_budget_enforcement_drops_context():
    assembler = RAGContextAssembler(max_chars=600)

    query = "Test query"

    chunks = [
        "A" * 300,
        "B" * 300,
        "C" * 300,
    ]

    prompt = assembler.assemble_prompt(
        query=query,
        context_chunks=chunks,
    )

    assert len(prompt) <= assembler.max_chars

    # At least one chunk must be omitted
    assert (
        "[Document 1]" not in prompt
        or "[Document 2]" not in prompt
        or "[Document 3]" not in prompt
    )


def test_empty_chunks_are_ignored():
    assembler = RAGContextAssembler()

    prompt = assembler.assemble_prompt(
        query="test",
        context_chunks=["", " ", "\n"],
    )

    assert "[No Context Included]" in prompt


def test_unicode_context_support():
    assembler = RAGContextAssembler(max_chars=2000)

    query = "量子化とは何ですか？"

    chunks = ["量子化はモデルサイズを削減する技術です。"]

    prompt = assembler.assemble_prompt(
        query=query,
        context_chunks=chunks,
    )

    assert "量子化" in prompt


def test_deterministic_output():
    assembler = RAGContextAssembler()

    query = "What is RAG?"

    chunks = ["RAG stands for Retrieval Augmented Generation."]

    prompt_1 = assembler.assemble_prompt(
        query,
        chunks,
    )

    prompt_2 = assembler.assemble_prompt(
        query,
        chunks,
    )

    assert prompt_1 == prompt_2


def test_too_small_budget_rejected():
    assembler = RAGContextAssembler(max_chars=100)

    with pytest.raises(
        ValueError,
        match="too small",
    ):
        assembler.assemble_prompt(
            query="hello",
            context_chunks=[],
        )
