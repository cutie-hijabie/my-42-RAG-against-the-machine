*This activity has been created as part of the 42 curriculum by hahel.*

# RAG Against the Machine

## Description

This 42 project answers questions about the vLLM 0.10.1 codebase and documentation. I built a lexical index over the supplied files and implemented BM25 to rank source passages. Qwen3-0.6B uses those passages to generate an answer. Search results can also be inspected and evaluated on their own.

## Instructions

Run the commands from the repository root, where `pyproject.toml` and `uv.lock` are located. Install uv, then sync the locked dependencies. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync --locked
```

The repository's `.python-version` selects Python 3.14. You can also install the dependencies with `make install`, which runs `uv sync`.

Place the 42-supplied vLLM corpus and question datasets under `data/`. The repository ignores this directory, so set up the inputs after checking out the project.

| Path | Contents |
| --- | --- |
| `data/raw/vllm-0.10.1/` | Supplied vLLM source tree, with its `docs/` and `vllm/` directories directly inside |
| `data/datasets/UnansweredQuestions/` | Question files for retrieval and answering |
| `data/datasets/AnsweredQuestions/` | Annotated reference files for evaluation |
| `data/processed/index.json` | Index created by the `index` command |
| `data/output/search_results/<DatasetScope>/` | Ranked source locations saved as JSON |
| `data/output/search_results_and_answer/<DatasetScope>/` | Saved source locations with generated answers |

Keep the supplied corpus at the expected path for grading. The evaluator compares file paths exactly. For generation, the default model and tokenizer are downloaded from Hugging Face on first use if they are missing from the local cache. Both answer commands accept `--model_name` with a local model directory. The `--device` default is `auto`, which selects CUDA when available and CPU otherwise.

Run `uv run python -m src` to list commands and `uv run python -m src index -- --help` to see the flags for `index`. For development checks, run `PATH="$PWD/.venv/bin:$PATH" make lint` after syncing. The Makefile prints lint findings and allows both checks to run even when one reports errors.

## System architecture

| Stage | Components | What happens |
| --- | --- | --- |
| Command line | [`__main__.py`](src/__main__.py), [`cli.py`](src/cli.py) | Python Fire exposes the `Cli` commands and routes arguments to each stage. |
| Indexing | [`corpus.py`](src/corpus.py), [`chunking.py`](src/chunking.py), [`indexing.py`](src/indexing.py) | `CorpusLoader` reads supported UTF-8 files, `Chunker` creates passages, and `Indexer` saves passages, term postings, and length statistics to JSON. |
| Retrieval | [`tokenization.py`](src/tokenization.py), [`retrieval.py`](src/retrieval.py) | `Retriever` uses the shared tokenizer and the saved postings to return the top-ranked source locations. |
| Answering | [`context.py`](src/context.py), [`generation.py`](src/generation.py) | Context helpers recover the original passages and build a prompt within the token budget. `Generator` runs Qwen and decodes the new tokens. |
| Evaluation | [`evaluation.py`](src/evaluation.py) | `Evaluator` pairs saved retrieval results with annotated questions and calculates source recall. The moulinette runs separately for official scoring. |
| JSON input and output | [`models.py`](src/models.py), [`datasets.py`](src/datasets.py) | Pydantic validates question datasets, indexes, and results. Batch results keep question IDs and input order. |

## Chunking strategy

- **Python (`.py`)** uses `ast.parse` to find top-level classes, functions, and async functions. A decorated definition starts at its first decorator. If parsing fails, the file goes through the size-based splitter.
- **Markdown (`.md`)** splits at hash-style headings outside backtick or tilde fences. **Plain text and reStructuredText (`.txt`, `.rst`)** also use blank lines. The `.rst` strategy recognizes headings underlined with `=` or `-`.
- Oversized sections are split into passages of at most `--max_chunk_size` characters. The splitter prefers a newline when one fits and overlaps consecutive pieces by up to `--overlap` characters. The defaults are 2,000 characters and 150 characters of requested overlap.

Each passage uses the original project-relative path and a half-open character range `[start, end)`. The corpus reader preserves newlines, so `text[start:end]` reconstructs the passage used at indexing time.

## Retrieval method

The same tokenizer is used for passages and queries. It lowercases words and keeps both whole identifiers and their parts. For example, `enable_lora` produces `enable_lora`, `enable`, and `lora`. Repeated occurrences in a passage count toward term frequency. Each distinct query token contributes to the score once.

The index stores postings by token, with the chunk ID and term frequency in each posting. BM25 scores the chunks found in those postings using term rarity across chunks, term-frequency saturation (`k1 = 1.5` by default), and length normalization (`b = 0.75` by default). Both parameters can be set when building the index.

A Markdown passage already matched by BM25 gets four additional points per distinct query token found in its corpus-relative file path after removing the extension. The final ranking uses descending score and ascending chunk ID for ties. Results are deduplicated by exact path and character range before returning up to `k` sources.

## Performance analysis

Recall@k is the share of annotated source ranges found among the top `k` retrieved results. A hit requires the same file path and a character-range intersection over union (IoU) of at least 0.05. The local evaluator averages per-question recall across questions with reference sources.

### Retrieval exam results

The Fedora moulinette run `2026-11-08_18-10-29` passed **all 4 checks** on the private datasets, with 100 questions each for docs and code.

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Docs | 60% | 74% | 80% | 82% | 80% |
| Code | 46% | 64% | 66% | 71% | 50% |

Docs recall@5 met the threshold exactly. Code recall@5 exceeded its threshold by 16 percentage points.

| Stage | Measured time | Exam limit |
| --- | ---: | ---: |
| Indexing | 16 s | 300 s |
| Warm retrieval, 200 questions | 12 s | 90 s |

Docs and code retrieval each took 6 seconds.

## Design decisions

- **Reusable index.** The index and result files are written atomically. Index loading compares the current corpus file paths, sizes, and modification times with the saved snapshot. Rebuild the index after editing or replacing the corpus.
- **Separate batch stages.** `search_dataset` loads one index for the batch. `answer_dataset` reads the saved source ranges and reuses one model across questions. Keep the source files available at their original paths when generating answers.
- **Bounded prompts.** The default 4,096-token budget reserves up to 512 tokens for generation. The prompt counts the instructions, question, and chat formatting. Passages are considered in rank order. If one does not fit, the builder can still include a later, smaller passage. Saved answer JSON keeps all retrieved source locations, including any passages omitted from the prompt.
- **Grounded generation.** The prompt tells Qwen to answer from the supplied evidence and say when the evidence is insufficient. The model runs with thinking and sampling disabled. If no passage fits, the generator returns an insufficient-evidence message.

## Challenges faced

- **Matching code identifiers with natural-language questions.** Keeping the full identifier and its split parts gives queries both ways to match the same passage.
- **Keeping Markdown topics together.** I used headings as section boundaries and ignored heading-like lines inside fenced code blocks. Long sections still need to be split to fit the 2,000-character limit.
- **Using the correct reference dataset.** Local evaluation rejected an `UnansweredQuestions` file because it had no source annotations. The reference input is the matching `AnsweredQuestions` file. Source paths also have to match the grader's project-relative paths exactly.
- **Fitting retrieved passages into Qwen's prompt.** The context builder counts actual chat-template tokens and reserves generation space before adding evidence.

## Example usage

### Build the index and try one question

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How do I enable LoRA in vLLM?" --k 5
uv run python -m src answer "How do I enable LoRA in vLLM?" --k 5 --device cpu
```

`search` prints the retrieved passages and saves `data/output/search_results/single/query.json`. `answer` retrieves evidence, prints the generated answer, and saves `data/output/search_results_and_answer/single/query.json`. Use `--save_directory` to keep separate runs. Use `--index_path` to select a different directory when building or reading the index.

### Search a dataset, evaluate it, and generate answers

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

The local evaluator needs the annotated dataset with the matching question IDs and text. Use the corresponding `dataset_code_public.json` files to evaluate code questions. Batch outputs keep the input filename. Scope output directories by dataset so that files with the same name from different runs stay separate.

For official scoring, use the supplied moulinette for your platform. Place the executable in the repository root as `moulinette` and make it executable before running this command.

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```

### Run the retrieval exam on Fedora

With the supplied exam scripts, private datasets, and Fedora moulinette in place, run this from the repository root.

```bash
bash exams/scripts/exam_retrieval.sh \
  --student-path . \
  --moulinette-path ./moulinette_pkg/moulinette-fedora
```

## Resources

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. [RAG paper](https://arxiv.org/abs/2005.11401)
- Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*. [BM25 paper](https://doi.org/10.1561/1500000019)
- Python's AST node types and parser. [Python `ast` documentation](https://docs.python.org/3/library/ast.html)
- Documentation for the indexed corpus version. [vLLM 0.10.1 documentation](https://docs.vllm.ai/en/v0.10.1/)
- Model behavior and the chat template used for generation. [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) · [Transformers chat templates](https://huggingface.co/docs/transformers/en/chat_templating)
- Dependency installation and environment syncing. [uv project documentation](https://docs.astral.sh/uv/concepts/projects/sync/)

### AI usage

I used ChatGPT to help plan the class-based structure, review and debug Python modules, suggest edge-case tests and BM25 comparisons, and draft docstrings and this README. In the application itself, Qwen3-0.6B generates the answers from the retrieved passages.
