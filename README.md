# RAG Against the Machine

**By Hayat Ahel** · Built at 42 Irbid

I built a Python command-line tool that answers questions about the vLLM codebase and documentation. Ask it how to enable LoRA, for example, and it searches the source files for relevant passages before passing them to a small local language model. The search results include file paths and character ranges, so you can go back to the exact text it found.

This is retrieval-augmented generation (RAG). I implemented the search and ranking with BM25 and used Qwen3-0.6B to turn the retrieved passages into an answer.

## How it works

| Stage | What it does |
| --- | --- |
| **Index** | Splits the vLLM 0.10.1 source and docs into passages. Python files use class and function boundaries, and Markdown files use heading boundaries. The index keeps the original location of each passage. |
| **Search** | Uses BM25 to rank passages. The tokenizer keeps both full identifiers and their parts, so a question about “enabling LoRA” can match an identifier like `enable_lora`. |
| **Answer** | Reads the original text from the returned file paths and ranges, then sends the passages that fit to Qwen3-0.6B. |
| **Evaluate** | Compares the retrieved locations with annotated source ranges and reports recall at different cutoffs. |

## How I built it

I worked one file at a time, starting with validation and data models. Then I built the corpus reader, chunker, index, and retriever in the order they depend on each other. That gave me a way to test each stage before connecting it to the next one.

I used classes for the parts with a clear job, like indexing, searching, and generating answers. I kept shared pieces like tokenization and JSON loading as functions. When I connected everything through the CLI, I made it possible to save a batch of search results and use the same results for evaluation or answer generation.

## What I struggled with

**Getting the right docs into the top five.** This took the most debugging. I compared missed questions with their expected source ranges and checked whether a matching passage was ever indexed. If it was there, I looked at which passages had ranked above it. One change I kept was a score boost for Markdown paths that share words with the question. A path can name the topic even when a passage explains it in different words.

**Tracking an evaluation error back to the input.** When I tested from a fresh `data/` directory, I kept getting `Reference question must be annotated`. I had given the evaluator the same question file I used for search. The fix was to pass the matching annotated file, which includes the source paths and ranges the evaluator needs to score the results.

**Fitting the retrieved text into the model’s context.** I capped chunks by character count, but the model’s limit is measured in tokens. I counted the formatted prompt with Qwen’s tokenizer, including the instructions and question, and set aside 512 of the 4,096 tokens for the answer.

## Results

The official 42 retrieval exam passed **4/4 checks** on Fedora. Recall@5 measures how much of the annotated evidence appears in the first five results.

| Held-out questions | Recall@5 | Required |
| --- | ---: | ---: |
| Documentation | **80%** | 80% |
| Code | **66%** | 50% |

In that run, indexing took 16 seconds, and retrieval took 12 seconds for 200 questions with the index already built. These scores measure how well search found the expected sources. Generated answers can be checked separately against those passages.

## Try it locally

From this repository’s root, with `uv` and Git installed, run the commands below. The project uses Python 3.14, and the example indexes the upstream vLLM 0.10.1 source tree. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) · [vLLM 0.10.1 documentation](https://docs.vllm.ai/en/v0.10.1/)

```bash
uv sync --locked
mkdir -p data/raw
git clone --depth 1 --branch v0.10.1 https://github.com/vllm-project/vllm.git data/raw/vllm-0.10.1

uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How do I enable LoRA in vLLM?" --k 5
uv run python -m src answer "How do I enable LoRA in vLLM?" --k 5 --device cpu
```

You can run `search` on its own. The first `answer` run downloads the model and tokenizer if they are missing from the local cache. The exam scores above came from the datasets supplied by 42.

## Resources

- [RAG paper](https://arxiv.org/abs/2005.11401)
- [BM25 paper](https://doi.org/10.1561/1500000019)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)

### AI usage

I wrote the implementation module by module. I used ChatGPT to discuss the design, review and debug code, suggest tests, and help with docstrings and this README.
