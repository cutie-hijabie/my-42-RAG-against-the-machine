from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.models import PreparedContext


class Generator:
    """Generate answer text from evidence prepared for a question.

    Load a pretrained tokenizer and causal language model once per
    instance. Reuse them to tokenize prepared prompts, check token
    budgets, and decode the model's generated answer text.

    Attributes:
        model_name: Pretrained model name or local model path.
        device: Selected device, either ``cpu`` or ``cuda``.
        context_limit: Total token budget for input and output.
        max_new_tokens: Maximum number of tokens to generate per answer.
        tokenizer: Pretrained tokenizer used for prompts and output.
        model: Pretrained causal language model used for inference.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
        context_limit: int = 4096,
        max_new_tokens: int = 512,
    ) -> None:
        """Validate generation settings and load the model and tokenizer.

        Select CUDA when available if the device is ``auto``.
        Otherwise, use the CPU. Load the model in 32-bit floating-point
        precision, move it to the selected device, and set it to
        evaluation mode. Check the context budget against the model's
        configured maximum number of positions.

        Args:
            model_name: Pretrained model name or local model path.
                Defaults to ``Qwen/Qwen3-0.6B``.
            device: Device selection, either ``auto``, ``cpu``,
                or ``cuda``.
            context_limit: Positive token budget for the prompt and
                generated answer combined.
            max_new_tokens: Positive number of tokens to reserve for
                generation. Must be smaller than ``context_limit``.

        Raises:
            ValueError: If a token limit or device selection is
                invalid, explicitly requested CUDA is unavailable, or
                the context budget exceeds the model's supported limit.
            RuntimeError: If the tokenizer or model cannot be loaded,
                or the model cannot be set up on the selected device.
        """
        if (
            not isinstance(context_limit, int)
            or context_limit < 1
            or isinstance(context_limit, bool)
        ):
            raise ValueError(
                "context_limit must be a positive integer; booleans are not"
                " allowed."
            )

        if (
            not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
            or isinstance(max_new_tokens, bool)
        ):
            raise ValueError(
                "max_new_tokens must be a positive integer; booleans are not"
                " allowed."
            )

        if not max_new_tokens < context_limit:
            raise ValueError("max_new_tokens must be less than context_limit.")

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        elif device == "cpu":
            pass

        elif device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is not available.")

        else:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")

        self.model_name = model_name
        self.device = device
        self.context_limit = context_limit
        self.max_new_tokens = max_new_tokens

        model_dtype = torch.float32
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=model_dtype,
            )
            self.model.to(self.device)
            self.model.eval()
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            raise RuntimeError(
                f"Could not load model '{self.model_name}' "
                f"on device '{self.device}': {error}"
            ) from error

        model_limit = self.model.config.max_position_embeddings

        if self.context_limit > model_limit:
            raise ValueError(
                f"context_limit ({self.context_limit}) exceeds the model's "
                f"supported limit ({model_limit})."
            )

    def generate_answer(self, prepared: PreparedContext) -> str:
        """Generate and decode an answer from a prepared context.

        Return an insufficient-evidence message when the context has
        no evidence. Otherwise, tokenize the prompt and verify that
        its length matches the recorded count and leaves enough room
        for the reserved output tokens.

        Run generation in inference mode with sampling disabled.
        Decode only the newly generated tokens, skip special tokens,
        and strip surrounding whitespace.

        Args:
            prepared: Formatted prompt, recorded input token count,
                and evidence metadata.

        Returns:
            The generated answer text or the insufficient-evidence
            message when no evidence was included.

        Raises:
            ValueError: If the actual prompt token count differs
                from the recorded count or exceeds the input budget.
            RuntimeError: If tokenization, generation, or decoding
                fails, or the decoded answer is empty.
        """

        if not prepared.has_evidence:
            return (
                "I could not find relevant evidence in the supplied sources."
            )

        try:
            inputs = self.tokenizer(
                prepared.prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            raise RuntimeError(
                f"Could not tokenize the prepared prompt: {error}"
            ) from error

        input_length = inputs["input_ids"].shape[1]

        if input_length != prepared.input_token_count:
            raise ValueError(
                f"Input token count ({input_length}) does not match "
                f"prepared count ({prepared.input_token_count})."
            )

        if input_length + self.max_new_tokens > self.context_limit:
            raise ValueError(
                "Input tokens plus max_new_tokens exceed context_limit."
            )

        try:
            inputs = inputs.to(self.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                )

        except (ValueError, TypeError, RuntimeError) as error:
            raise RuntimeError(
                f"Could not generate an answer on '{self.device}': {error}"
            ) from error

        output = output_ids[0, input_length:]
        try:
            readable_output = self.tokenizer.decode(
                output, skip_special_tokens=True
            ).strip()

        except (ValueError, TypeError, RuntimeError) as error:
            raise RuntimeError(
                f"Could not decode the generated answer: {error}"
            ) from error

        if not readable_output:
            raise RuntimeError("The model generated an empty answer.")

        return str(readable_output)
