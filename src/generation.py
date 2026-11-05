from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.models import PreparedContext


class Generator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
        context_limit: int = 4096,
        max_new_tokens: int = 512,
    ) -> None:
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

        return readable_output
