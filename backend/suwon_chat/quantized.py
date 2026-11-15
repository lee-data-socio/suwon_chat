"""GPU 없이 돌리는 실시간 예측. 양자화한 Bllossom 8B(GGUF)를 llama.cpp로 CPU에서 돌린다.

    python -m suwon_chat.quantized --download      # 양자화 모델(약 4.9GB)을 models/ 로 받는다
    PREDICT_BACKEND=gguf python -m uvicorn suwon_chat.api:app --port 8000
    python -m suwon_chat.live_check --backend gguf # 저장된 예측과 얼마나 다른지 잰다

계산은 `live.py`와 같다(프롬프트, 보기 글자 토큰, softmax, 정규화). 다른 점은 가중치가 4비트로 줄어 있다는 것이다.
그래서 값이 저장된 예측과 조금 다르다. 테스트와 시연 연습용이다. 보고서의 숫자와 나란히 놓을 때는 조회 버전을 쓴다.
어댑터는 `scripts/convert_adapters_gguf.py`로 바꿔 둔 `adapters/exp*/adapter.gguf`를 쓴다.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from pathlib import Path

from . import config
from .live import BASE, InferenceBackend, LiveError
from .prompt import MAX_OPTIONS, letter
from .store import Store

log = logging.getLogger("suwon_chat.quantized")


class QuantizedBackend(InferenceBackend):
    def __init__(self, store: Store, model_path: Path | None = None, adapter_files: dict[str, Path] | None = None,
                 n_threads: int | None = None, n_ctx: int = 1024):
        self._init_common(store)
        self.model_path = Path(model_path or config.GGUF_MODEL)
        self.adapter_files = adapter_files or {k: config.ADAPTER_DIR / k / "adapter.gguf" for k in config.ADAPTERS}
        self.adapter_names = list(self.adapter_files)
        self.n_threads = n_threads or config.GGUF_THREADS or max(1, (os.cpu_count() or 2) // 2)
        self.n_ctx = n_ctx
        self._llm = None
        self._adapters: dict[str, object] = {}
        self._letters: list[tuple[int, int]] = []
        self.load_seconds: float | None = None

    def load(self) -> None:
        if self._llm is not None:
            return
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as e:
            raise LiveError("양자화 모델을 돌리려면 llama-cpp-python이 필요하다. README의 '양자화 버전' 절에 설치 명령이 있다.") from e
        if not self.model_path.is_file():
            raise LiveError(f"양자화 모델 파일이 없다: {self.model_path}\n`python -m suwon_chat.quantized --download`로 받는다 (약 4.9GB).")
        for name, path in self.adapter_files.items():
            if not Path(path).is_file():
                raise LiveError(f"어댑터 {name}의 GGUF 파일이 없다: {path}\n`python scripts/convert_adapters_gguf.py`로 만든다.")
        size = self.model_path.stat().st_size / 1e9
        try:
            free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        except (AttributeError, ValueError, OSError):
            free = None
        if free is not None and free < size * 0.9:
            log.warning("RAM이 빠듯하다: 모델 파일 %.1f GB, 지금 쓸 수 있는 RAM %.1f GB. 느려지거나 멈출 수 있다.", size, free)

        t0 = time.time()
        log.info("양자화 모델을 올린다: %s (%.1f GB, CPU 스레드 %d개)", self.model_path.name, size, self.n_threads)
        try:
            llm = Llama(model_path=str(self.model_path), n_ctx=self.n_ctx, n_batch=512, n_threads=self.n_threads,
                        n_threads_batch=self.n_threads, n_gpu_layers=0, logits_all=False, verbose=False)
            for name, path in self.adapter_files.items():
                handle = llama_cpp.llama_adapter_lora_init(llm._model.model, str(path).encode("utf-8"))
                if not handle:
                    raise LiveError(f"어댑터 {name}을 올리지 못했다: {path} (기본 모델과 맞지 않는 어댑터일 수 있다)")
                self._adapters[name] = handle
        except LiveError:
            raise
        except Exception as e:
            raise LiveError(f"양자화 모델을 올리지 못했다: {type(e).__name__}: {str(e)[:300]}") from e
        last = lambda text: llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)[-1]
        self._letters = [(last("Answer:" + letter(i)), last("Answer: " + letter(i))) for i in range(MAX_OPTIONS)]
        self._llama_cpp, self._llm = llama_cpp, llm
        self.load_seconds = round(time.time() - t0, 1)
        log.info("양자화 모델을 올렸다 (%.1f초). 어댑터: %s", self.load_seconds, ", ".join(self.adapter_names))

    def describe(self) -> dict:
        return {"engine": "llama.cpp", "quantized": True, "base_model": self.model_path.name, "revision": None,
                "device": "cpu", "dtype": _quant_name(self.model_path.name), "four_bit": True, "batch_size": 1,
                "adapters": self.adapter_names, "load_seconds": self.load_seconds, "threads": self.n_threads}

    def _activate(self, adapter: str) -> None:
        cpp, ctx = self._llama_cpp, self._llm._ctx.ctx
        if adapter == BASE:
            result = cpp.llama_set_adapters_lora(ctx, None, 0, None)
        else:
            handles = (cpp.llama_adapter_lora_p_ctypes * 1)(self._adapters[adapter])
            scales = (ctypes.c_float * 1)(1.0)
            result = cpp.llama_set_adapters_lora(ctx, handles, 1, scales)
        if result:
            raise LiveError(f"어댑터를 바꿔 끼우지 못했다: {adapter}")

    def tokens(self, prompt: str) -> list[int]:
        return self._llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=False)

    def _forward_with(self, adapter: str, prompts: list[str], n_options: int) -> list[list[float]]:
        import numpy as np
        self._activate(adapter)
        llm, n_vocab, out = self._llm, self._llm.n_vocab(), []
        for prompt in prompts:
            tokens = self.tokens(prompt)
            if len(tokens) > self.n_ctx:
                raise LiveError(f"프롬프트가 너무 길다: {len(tokens)}토큰 (한도 {self.n_ctx})")
            llm.reset()                       # 다른 어댑터로 계산해 둔 앞부분을 다시 쓰지 않게 처음부터 돌린다
            llm._ctx.kv_cache_clear()
            llm.eval(tokens)
            logits = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(n_vocab,)).astype(np.float64)
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            p = [float(probs[a] + probs[b]) for a, b in self._letters[:n_options]]
            total = sum(p)
            if not total > 0:
                raise LiveError("모델이 보기 글자에 확률을 주지 않았다")
            out.append([round(x / total, 5) for x in p])
        return out


def _quant_name(filename: str) -> str:
    import re
    m = re.search(r"(Q\d+_[A-Z0-9_]+|Q\d+)", filename, re.I)
    return (m.group(1).upper() if m else "GGUF") + " 양자화"


def download() -> Path:
    from huggingface_hub import hf_hub_download
    config.GGUF_MODEL.parent.mkdir(parents=True, exist_ok=True)
    print(f"{config.GGUF_REPO} 에서 {config.GGUF_FILE} 을 받는다 (약 4.9GB)…", flush=True)
    path = hf_hub_download(config.GGUF_REPO, config.GGUF_FILE, local_dir=str(config.GGUF_MODEL.parent))
    print("받았다:", path)
    return Path(path)


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    else:
        sys.exit(__doc__)
