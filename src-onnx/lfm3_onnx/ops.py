"""ORT sessions and caches, adapted from Liquid4All/onnx-export. See NOTICE."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

ONNX_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16,
    "tensor(int64)": np.int64, "tensor(int32)": np.int32,
}


def default_ep(force_cpu=False):
    import onnxruntime as ort
    return "cuda" if not force_cpu and "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"


def load_session(path, ep="cpu", tf32=False):
    import onnxruntime as ort

    if not path.is_file():
        raise FileNotFoundError(f"Missing graph: {path}. Use --download to fetch this bundle.")
    options = ort.SessionOptions()
    options.add_session_config_entry("session.set_denormal_as_zero", "1")
    # Five graphs share the process. Keep separate thread pools small.
    options.intra_op_num_threads = 2
    providers = ["CPUExecutionProvider"]
    if ep == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA requires the gpu extra: uv run --project src-onnx --extra gpu serve --device cuda")
        preload = getattr(ort, "preload_dlls", None)
        if preload:
            preload()
        providers.insert(0, ("CUDAExecutionProvider", {
            "sdpa_kernel": "3", "cudnn_conv_algo_search": "HEURISTIC",
            "use_tf32": "1" if tf32 else "0",
        }))
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    if ep == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDA provider failed to load. Check ONNX Runtime's CUDA/cuDNN requirements.")
    return session


def initialize_cache(session: ort.InferenceSession, batch_size: int = 1) -> dict:
    """Empty conv and KV caches (past_* inputs), typed as the graph declares them."""
    cache = {}
    for inp in session.get_inputs():
        if not inp.name.startswith("past"):
            continue
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        for i, d in enumerate(inp.shape):
            if isinstance(d, str) and "sequence" in d.lower():
                shape[i] = 0
        shape[0] = batch_size
        dtype = ONNX_TYPE_TO_NUMPY.get(inp.type, np.float32)
        cache[inp.name] = np.zeros(shape, dtype=dtype)
    return cache


def cache_input_name(output: str) -> str | None:
    """present_conv.N -> past_conv.N, present.N.key -> past_key_values.N.key."""
    if output.startswith("present_conv."):
        return output.replace("present_conv.", "past_conv.", 1)
    if output.startswith("present."):
        return output.replace("present.", "past_key_values.", 1)
    return None


class CachedDecoder:
    """The decoder steps of one generation, with the caches bound in place between steps.

    Caches stay where their consumers run: on the GPU in CUDA sessions, except float32 KV, whose
    GroupQueryAttention has no CUDA kernel and runs on the CPU. With a capacity (and KV that
    feeds GroupQueryAttention directly), each KV cache is one [B, kv_heads, capacity, head_dim]
    buffer bound as both past and present, so attention appends in place; on CUDA that also
    enables its XQA decode kernel.
    """

    def __init__(self, session: ort.InferenceSession, capacity: int | None = None):
        import onnxruntime as ort

        self.session = session
        self.capacity = capacity
        cuda = "CUDAExecutionProvider" in session.get_providers()
        self.outputs = [o.name for o in session.get_outputs()]
        self.inputs = {i.name for i in session.get_inputs()}
        self.cache, self.shared = {}, set()
        for name, array in initialize_cache(session).items():
            kv = name.startswith("past_key_values.")
            device = "cuda" if cuda and not (kv and array.dtype == np.float32) else "cpu"
            if kv and capacity:
                array = np.zeros((*array.shape[:2], capacity, array.shape[3]), dtype=array.dtype)
                self.shared.add(name)
            self.cache[name] = (ort.OrtValue.ortvalue_from_numpy(array, device, 0), device)
        self.past = 0

    def step(self, ids: list[int], **inputs: np.ndarray) -> np.ndarray:
        """logits of ids after the tokens so far; inputs are the graph's other inputs."""
        n = len(ids)
        if self.capacity and self.past + n > self.capacity:
            raise ValueError(f"{self.past + n} tokens exceed the cache capacity {self.capacity}")
        binding = self.session.io_binding()
        binding.bind_cpu_input("input_ids", np.array([ids], dtype=np.int64))
        binding.bind_cpu_input("attention_mask", np.ones((1, self.past + n), dtype=np.int64))
        positions = np.arange(self.past, self.past + n, dtype=np.int64)[None, :]
        binding.bind_cpu_input("position_ids", positions)
        for name, value in inputs.items():
            if name in self.inputs:
                binding.bind_cpu_input(name, value)
        for name, (value, _) in self.cache.items():
            binding.bind_ortvalue_input(name, value)
        for name in self.outputs:
            past = cache_input_name(name)
            if past in self.shared:
                binding.bind_ortvalue_output(name, self.cache[past][0])
            elif past in self.cache:
                binding.bind_output(name, self.cache[past][1])
            else:
                binding.bind_output(name, "cpu")
        self.session.run_with_iobinding(binding)
        values = binding.get_outputs()
        for name, value in zip(self.outputs, values, strict=True):
            past = cache_input_name(name)
            if past in self.cache and past not in self.shared:
                self.cache[past] = (value, self.cache[past][1])
        self.past += n
        # Every output of the step by name (OrtValues), e.g. an audio decoder's hidden_states
        self.last_outputs = dict(zip(self.outputs, values, strict=True))
        return self.last_outputs["logits"].numpy()

