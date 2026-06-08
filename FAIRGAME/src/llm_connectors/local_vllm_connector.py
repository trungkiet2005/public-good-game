"""
Local LLM Connector using vLLM for offline GPU inference.
Designed for Kaggle environments with no internet access.

Usage:
    The model must be pre-downloaded and available as a local path.
    On Kaggle, this means uploading the model weights as a dataset
    and pointing MODEL_PATH to /kaggle/input/<dataset-name>/<model-folder>.
"""

from src.llm_connectors.abstract_connector import AbstractConnector

# Global singleton to avoid loading the model multiple times
_GLOBAL_LLM = None
_GLOBAL_TOKENIZER = None
_GLOBAL_SAMPLING_PARAMS = None
_GLOBAL_ENGINE_TYPE = None  # "vllm" or "transformers"


def _init_vllm_engine(model_path: str, max_model_len: int = 4096,
                      temperature: float = 1.0, max_tokens: int = 1024,
                      gpu_memory_utilization: float = 0.90,
                      tensor_parallel_size: int = 1):
    """Initialize vLLM engine (preferred for high throughput)."""
    from vllm import LLM, SamplingParams

    global _GLOBAL_LLM, _GLOBAL_SAMPLING_PARAMS, _GLOBAL_ENGINE_TYPE

    _GLOBAL_SAMPLING_PARAMS = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=5,  # Capture top-5 token logprobs for XAI analysis
    )
    _GLOBAL_LLM = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,  # Avoid CUDA graph issues on some setups
    )
    _GLOBAL_ENGINE_TYPE = "vllm"
    print(f"[LocalVLLM] Loaded model from {model_path} with vLLM engine")


def _init_transformers_engine(model_path: str, temperature: float = 1.0,
                               max_tokens: int = 1024):
    """Fallback: Initialize HuggingFace Transformers pipeline.

    Optimizations applied:
      - Flash-Attention 2 (fallback to SDPA if package missing)
      - eval() + cudnn benchmark for fixed-shape kernels
      - Left-padding tokenizer (cần khi muốn batch sau này)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    global _GLOBAL_LLM, _GLOBAL_TOKENIZER, _GLOBAL_SAMPLING_PARAMS, _GLOBAL_ENGINE_TYPE

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if _GLOBAL_TOKENIZER.pad_token_id is None:
        _GLOBAL_TOKENIZER.pad_token_id = _GLOBAL_TOKENIZER.eos_token_id

    common_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    try:
        _GLOBAL_LLM = AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="flash_attention_2", **common_kwargs
        )
        print("[LocalVLLM] attn=flash_attention_2")
    except (ImportError, ValueError, RuntimeError) as e:
        _GLOBAL_LLM = AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="sdpa", **common_kwargs
        )
        print(f"[LocalVLLM] flash-attn unavailable ({type(e).__name__}); attn=sdpa")

    _GLOBAL_LLM.eval()

    _GLOBAL_SAMPLING_PARAMS = {
        "temperature": temperature,
        "max_new_tokens": max_tokens,
    }
    _GLOBAL_ENGINE_TYPE = "transformers"
    print(f"[LocalVLLM] Loaded model from {model_path} with Transformers engine")


def init_local_llm(model_path: str, engine: str = "vllm", **kwargs):
    """
    Initialize the local LLM engine. Call this ONCE at startup.

    Args:
        model_path: Path to the local model directory.
        engine: "vllm" (recommended) or "transformers" (fallback).
        **kwargs: Additional arguments passed to the engine init function.
    """
    global _GLOBAL_LLM
    if _GLOBAL_LLM is not None:
        print("[LocalVLLM] Engine already initialized, skipping.")
        return

    if engine == "vllm":
        _init_vllm_engine(model_path, **kwargs)
    elif engine == "transformers":
        _init_transformers_engine(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown engine type: {engine}. Use 'vllm' or 'transformers'.")


def _generate_vllm(prompt: str) -> str:
    """Generate text using vLLM engine."""
    outputs = _GLOBAL_LLM.generate([prompt], _GLOBAL_SAMPLING_PARAMS)
    return outputs[0].outputs[0].text.strip()


def _generate_vllm_with_details(prompt: str) -> dict:
    """Generate text using vLLM with XAI details (logprobs, top alternatives).

    Returns:
        dict with keys: text, logprobs, top_alternatives, cumulative_logprob
    """
    outputs = _GLOBAL_LLM.generate([prompt], _GLOBAL_SAMPLING_PARAMS)
    output = outputs[0].outputs[0]

    result = {
        "text": output.text.strip(),
        "cumulative_logprob": output.cumulative_logprob,
        "token_logprobs": [],
        "top_alternatives": [],
    }

    if output.logprobs:
        for token_logprob_dict in output.logprobs:
            sorted_entries = sorted(
                token_logprob_dict.values(),
                key=lambda x: x.logprob,
                reverse=True
            )
            if sorted_entries:
                top = sorted_entries[0]
                result["token_logprobs"].append({
                    "token": top.decoded_token,
                    "logprob": top.logprob,
                })
                result["top_alternatives"].append([
                    {"token": e.decoded_token, "logprob": e.logprob}
                    for e in sorted_entries[:5]
                ])

    return result


def _format_prompt_for_model(prompt: str) -> str:
    """Apply chat template if model has one, else return prompt as-is."""
    if hasattr(_GLOBAL_TOKENIZER, "chat_template") and _GLOBAL_TOKENIZER.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return _GLOBAL_TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt


def _generate_transformers(prompt: str) -> str:
    """Generate text using Transformers engine (single prompt)."""
    import torch

    text = _format_prompt_for_model(prompt)

    inputs = _GLOBAL_TOKENIZER(text, return_tensors="pt").to(_GLOBAL_LLM.device)
    with torch.inference_mode():
        output_ids = _GLOBAL_LLM.generate(
            **inputs,
            do_sample=_GLOBAL_SAMPLING_PARAMS["temperature"] > 0,
            temperature=_GLOBAL_SAMPLING_PARAMS["temperature"] if _GLOBAL_SAMPLING_PARAMS["temperature"] > 0 else None,
            max_new_tokens=_GLOBAL_SAMPLING_PARAMS["max_new_tokens"],
            use_cache=True,
            pad_token_id=_GLOBAL_TOKENIZER.pad_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _GLOBAL_TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_transformers_batch(prompts: list) -> list:
    """Generate text for many prompts in a single batched forward pass.

    Uses left-padding so all sequences end at the same position; new tokens
    are decoded by slicing past the input length.
    """
    import torch

    texts = [_format_prompt_for_model(p) for p in prompts]
    inputs = _GLOBAL_TOKENIZER(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(_GLOBAL_LLM.device)

    with torch.inference_mode():
        output_ids = _GLOBAL_LLM.generate(
            **inputs,
            do_sample=_GLOBAL_SAMPLING_PARAMS["temperature"] > 0,
            temperature=_GLOBAL_SAMPLING_PARAMS["temperature"] if _GLOBAL_SAMPLING_PARAMS["temperature"] > 0 else None,
            max_new_tokens=_GLOBAL_SAMPLING_PARAMS["max_new_tokens"],
            use_cache=True,
            pad_token_id=_GLOBAL_TOKENIZER.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(len(texts)):
        new_tokens = output_ids[i][input_len:]
        results.append(
            _GLOBAL_TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
        )
    return results


def _generate_vllm_batch(prompts: list) -> list:
    """Batched generation via vLLM (single .generate() call handles batching)."""
    outputs = _GLOBAL_LLM.generate(prompts, _GLOBAL_SAMPLING_PARAMS)
    return [o.outputs[0].text.strip() for o in outputs]


def send_prompts_global(prompts: list, batch_size: int = 0) -> list:
    """Module-level batched send. Used by batch_runner without instantiating
    a connector. ``batch_size=0`` runs the whole list as one batch.
    """
    if _GLOBAL_LLM is None:
        raise RuntimeError("Local LLM not initialized. Call init_local_llm() first.")
    if not prompts:
        return []
    if batch_size and batch_size > 0:
        results = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            if _GLOBAL_ENGINE_TYPE == "vllm":
                results.extend(_generate_vllm_batch(chunk))
            else:
                results.extend(_generate_transformers_batch(chunk))
        return results
    if _GLOBAL_ENGINE_TYPE == "vllm":
        return _generate_vllm_batch(prompts)
    return _generate_transformers_batch(prompts)


class LocalVLLMConnector(AbstractConnector):
    """
    Chat model connector for locally hosted LLMs via vLLM or Transformers.
    Must call init_local_llm() before using this connector.
    """

    def __init__(self, provider_model: str, temperature: float = 1.0):
        """
        Args:
            provider_model: The model identifier (used for logging/tracking).
            temperature: Sampling temperature (applied at init_local_llm level).
        """
        self.provider_model = provider_model
        self.temperature = temperature

        if _GLOBAL_LLM is None:
            raise RuntimeError(
                "Local LLM not initialized. Call init_local_llm(model_path) first."
            )

    def send_prompt(self, prompt: str) -> str:
        """
        Send a prompt to the locally loaded LLM and return the response.

        Args:
            prompt: The prompt text.

        Returns:
            The generated text response.
        """
        if _GLOBAL_ENGINE_TYPE == "vllm":
            return _generate_vllm(prompt)
        elif _GLOBAL_ENGINE_TYPE == "transformers":
            return _generate_transformers(prompt)
        else:
            raise RuntimeError(f"Unknown engine type: {_GLOBAL_ENGINE_TYPE}")

    def send_prompt_with_details(self, prompt: str) -> dict:
        """
        Send a prompt and return detailed output including logprobs for XAI analysis.
        Only available with vLLM engine.

        Args:
            prompt: The prompt text.

        Returns:
            dict: Contains 'text', 'cumulative_logprob', 'token_logprobs', 'top_alternatives'.
                  Falls back to {'text': response} for transformers engine.
        """
        if _GLOBAL_ENGINE_TYPE == "vllm":
            return _generate_vllm_with_details(prompt)
        else:
            return {"text": self.send_prompt(prompt)}
