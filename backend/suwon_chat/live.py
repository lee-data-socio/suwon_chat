"""실시간 예측 백엔드. Bllossom 8B를 올리고 학습한 LoRA 어댑터를 갈아 끼우며 직접 추론한다.

정리본 `predict_local.py`와 같은 계산이다: 프롬프트 끝의 `Answer:` 다음 토큰 확률에서 보기 글자
("Answer:A"와 "Answer: A" 두 표기)의 확률을 더해 보기 수만큼 정규화한다.

    PREDICT_BACKEND=live python -m uvicorn suwon_chat.api:app --port 8000
    python -m suwon_chat.live_check          # 저장된 예측과 같은 값이 나오는지 확인 (PRD A6)

필요한 것: 모델 가중치 약 16GB(처음 한 번 내려받는다)와 그만큼의 GPU 메모리 또는 RAM.
`LIVE_4BIT=1`이면 4비트로 줄여 올린다(NVIDIA GPU, bitsandbytes 필요). 이때는 저장된 예측과 값이 조금 달라진다.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path

from . import config
from .prompt import MAX_OPTIONS, build_prompt, letter
from .store import Store

log = logging.getLogger("suwon_chat.live")
BASE = "__base__"      # 어댑터를 끈 학습 전 모델
CACHE_MAX = 5000       # 기억해 둘 (어댑터, 문항, 속성) 조합의 수. 목록의 문항 전부(약 3,600)가 들어간다


class LiveError(Exception):
    pass


DTYPES = ("bfloat16", "float16", "float32")
NEED_GB = {"bfloat16": 17.0, "float16": 17.0, "float32": 33.0}     # 8B 가중치를 올리는 데 드는 메모리 (대략)
NEED_GB_4BIT = 7.0
ADVICE = ("대처: GPU 메모리가 모자라면 LIVE_4BIT=1(NVIDIA GPU, `pip install bitsandbytes`. 값이 조금 달라진다), "
          "아니면 조회 버전(PREDICT_BACKEND=lookup)을 쓴다.")


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():      # Intel GPU
        return "xpu"
    return "cpu"


def _free_gb(device: str) -> float | None:
    """그 장치에서 지금 쓸 수 있는 메모리(GB). 알 수 없으면 None."""
    import torch
    try:
        if device == "cuda":
            return torch.cuda.mem_get_info()[0] / 1e9
        if device in ("cpu", "mps"):
            import os
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9     # 윈도우에서는 없다
    except (AttributeError, ValueError, OSError, RuntimeError):
        pass
    return None


class InferenceBackend:
    """모델을 직접 돌리는 백엔드의 공통 부분: 프롬프트 만들기, 어댑터별 실행, 결과 기억해 두기.

    하위 클래스는 `load()`와 `_forward_with(adapter, prompts, n_options)`를 채운다.
    """
    source = "live"
    store: Store
    adapter_names: list[str]

    def _init_common(self, store: Store) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._cache: dict[tuple, list[list[float]]] = {}

    def load(self) -> None:
        raise NotImplementedError

    def _forward_with(self, adapter: str, prompts: list[str], n_options: int) -> list[list[float]]:
        """어댑터 하나(BASE면 학습 전 모델)를 끼우고 프롬프트마다 보기 분포를 낸다."""
        raise NotImplementedError

    def _run(self, key: tuple, adapter: str, prompts: list[str], n_options: int) -> list[list[float]]:
        """어댑터 하나(또는 학습 전 모델)로 프롬프트들을 돌린다. 같은 요청은 다시 돌리지 않는다."""
        cache_key = (adapter, *key)
        if cache_key in self._cache:
            return self._cache[cache_key]
        self.load()
        with self._lock:          # 어댑터를 바꿔 끼우는 동안 다른 요청이 끼어들지 못하게 한다
            if cache_key not in self._cache:
                if adapter != BASE and adapter not in self.adapter_names:
                    raise LiveError(f"없는 어댑터: {adapter}")
                t0 = time.time()
                dists = self._forward_with(adapter, prompts, n_options)
                log.info("실시간 예측 %s × %d개 프롬프트: %.2f초", adapter, len(prompts), time.time() - t0)
                if len(self._cache) >= CACHE_MAX:          # 새 문항을 계속 물어도 끝없이 커지지 않게 한다
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = dists
        return self._cache[cache_key]

    def _predict(self, key: tuple, attribute: str, adapter: str, question: str, options: list[str]) -> dict:
        groups = self.store.groups[attribute]
        steer = self.store.steering[attribute]
        prompts = [build_prompt(steer, groups, g, question, options) for g in groups]
        lora = self._run(key, adapter, prompts, len(options))
        base = self._run(key, BASE, prompts, len(options))
        return {g: {"lora": lora[i], "base": base[i]} for i, g in enumerate(groups)}

    def predict(self, qkey: str, attribute: str, adapter: str) -> dict[str, dict[str, list[float]]]:
        q = self.store.questions[qkey]
        return self._predict((qkey, attribute), attribute, adapter, q.text, q.all_options)

    def predict_new(self, question: str, options: list[str], attribute: str, adapter: str) -> dict:
        """목록에 없는 새 문항. `options`는 끝의 무응답까지 포함한다."""
        digest = hashlib.sha256("\x1f".join([question, *options]).encode()).hexdigest()[:16]
        return self._predict(("new", digest, attribute), attribute, adapter, question, options)


class LiveBackend(InferenceBackend):
    """GPU에서 원래 정밀도(bfloat16)의 Bllossom 8B를 transformers + PEFT로 돌린다."""

    def __init__(self, store: Store, base_model: str | None = None, adapter_dirs: dict[str, Path] | None = None,
                 device: str | None = None, dtype: str | None = None, load_4bit: bool | None = None,
                 batch_size: int | None = None, revision: str | None = None):
        self._init_common(store)
        self.base_model = base_model or config.LIVE_BASE_MODEL
        # 허깅페이스에서 받을 때는 판을 고정한다. 저장소가 바뀌어도 어댑터를 학습한 가중치를 받는다
        self.revision = revision if revision is not None else (
            config.LIVE_BASE_REVISION if self.base_model == config.DEFAULT_BASE_MODEL else None)
        self.adapter_dirs = adapter_dirs or {k: config.ADAPTER_DIR / k for k in config.ADAPTERS}
        self.device = device or config.LIVE_DEVICE or None
        self.dtype = dtype or config.LIVE_DTYPE or None
        self.load_4bit = config.LIVE_4BIT if load_4bit is None else load_4bit
        self.batch_size = batch_size or config.LIVE_BATCH or None
        self.adapter_names = list(self.adapter_dirs)
        self._model = self._tok = None
        self._letters: list[tuple[int, int]] = []
        self.load_seconds: float | None = None

    # ---- 모델 올리기 --------------------------------------------------------
    def _check_before_loading(self) -> None:
        """16GB를 읽기 전에 알 수 있는 문제는 먼저 알린다."""
        import shutil
        import torch
        for name, path in self.adapter_dirs.items():
            if not (Path(path) / "adapter_config.json").is_file():
                raise LiveError(f"어댑터 {name}이 없다: {path}")
        auto = self.device is None
        self.device = self.device or pick_device()
        if self.device not in ("cuda", "mps", "xpu", "cpu"):
            raise LiveError(f"LIVE_DEVICE는 cuda, mps, xpu, cpu 가운데 하나여야 한다: {self.device!r}")
        available = {"cuda": torch.cuda.is_available(),
                     "mps": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
                     "xpu": getattr(torch, "xpu", None) is not None and torch.xpu.is_available(), "cpu": True}
        nvidia_without_cuda = torch.version.cuda is None and shutil.which("nvidia-smi") is not None
        hint = (" 이 PC에는 NVIDIA GPU가 있는데 지금 깔린 torch가 CPU용이다. "
                "pytorch.org의 설치 명령으로 CUDA용 torch를 다시 깐다.") if nvidia_without_cuda else ""
        if not available[self.device]:
            raise LiveError(f"{self.device} 장치를 쓸 수 없다.{hint}")
        # 저장된 예측은 bfloat16으로 계산한 값이다. 같은 값을 내려면 어느 장치에서든 bfloat16을 쓴다
        self.dtype = self.dtype or "bfloat16"
        if self.dtype not in DTYPES:
            raise LiveError(f"LIVE_DTYPE은 {', '.join(DTYPES)} 가운데 하나여야 한다: {self.dtype!r}")
        if self.load_4bit and self.device != "cuda":
            raise LiveError("LIVE_4BIT는 NVIDIA GPU(cuda)에서만 쓸 수 있다.")
        if self.device == "cpu":
            log.warning("GPU를 찾지 못해 CPU로 돌린다%s. 8B 모델이면 RAM이 %d GB쯤 필요하고, 한 장면에 몇 분이 걸릴 수 있다.%s",
                        "" if auto else " (LIVE_DEVICE=cpu)", NEED_GB[self.dtype], hint)
        need = self._need_gb()
        free = _free_gb(self.device)
        if need is not None and free is not None and free < need * 0.9:
            where = "GPU 메모리" if self.device == "cuda" else "RAM"
            raise LiveError(f"{where}가 모자란다: 약 {need:.0f} GB가 필요한데 지금 쓸 수 있는 것은 {free:.1f} GB다. {ADVICE}")

    def _need_gb(self) -> float | None:
        """모델을 올리는 데 들 메모리(GB)의 어림값. 알 수 없으면 None."""
        if self.base_model == config.DEFAULT_BASE_MODEL:
            return NEED_GB_4BIT if self.load_4bit else NEED_GB[self.dtype]
        folder = Path(self.base_model)
        if folder.is_dir():          # 따로 받아 둔 모델: 가중치 파일 크기로 어림한다
            size = sum(f.stat().st_size for f in folder.glob("*.safetensors")) + sum(f.stat().st_size for f in folder.glob("*.bin"))
            return size / 1e9 * 1.1 if size else None
        return None

    def load(self) -> None:
        """기본 모델과 어댑터 세 개를 올린다. 실패하면 이유를 적어 LiveError를 낸다."""
        if self._model is not None:
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise LiveError(f"실시간 예측에 필요한 패키지가 없다 ({e.name}). `pip install -r requirements.txt`를 돌린다.") from e
        self._check_before_loading()

        t0 = time.time()
        # device_map으로 그 장치에 바로 올린다. RAM에 한 번 다 올렸다가 옮기면 메모리가 두 배로 든다
        kwargs: dict = {"dtype": getattr(torch, self.dtype), "device_map": {"": self.device}}
        if self.load_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise LiveError("LIVE_4BIT에는 bitsandbytes가 필요하다: `pip install bitsandbytes`") from e
            kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.bfloat16),
                      "device_map": {"": 0}}
        if self.revision:
            kwargs["revision"] = self.revision
        log.info("기본 모델을 올린다: %s (%s, %s%s)", self.base_model, self.device, self.dtype,
                 ", 4비트" if self.load_4bit else "")
        try:
            tok = AutoTokenizer.from_pretrained(self.base_model, **({"revision": self.revision} if self.revision else {}))
            model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
            names = list(self.adapter_dirs)
            model = PeftModel.from_pretrained(model, str(self.adapter_dirs[names[0]]), adapter_name=names[0])
            for name in names[1:]:
                model.load_adapter(str(self.adapter_dirs[name]), adapter_name=name)
            model.eval()
        except LiveError:
            raise
        except Exception as e:
            out_of_memory = "out of memory" in str(e).lower() or type(e).__name__ in ("OutOfMemoryError", "MemoryError")
            advice = f" 메모리가 모자란다. {ADVICE}" if out_of_memory else ""
            raise LiveError(f"모델을 올리지 못했다: {type(e).__name__}: {str(e)[:300]}{advice}") from e

        # 여러 프롬프트를 한 번에 돌릴 때 마지막 토큰 자리를 맞추려고 왼쪽을 채운다
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        # "Answer:A"와 "Answer: A"의 마지막 토큰 둘 다 센다 (SubPOP run_inference.py와 같은 방식)
        self._letters = [(tok.encode("Answer:" + letter(i), add_special_tokens=False)[-1],
                          tok.encode("Answer: " + letter(i), add_special_tokens=False)[-1])
                         for i in range(MAX_OPTIONS)]
        self._tok, self._model = tok, model
        self.batch_size = self.batch_size or (8 if self.device == "cuda" else 1)
        self.load_seconds = round(time.time() - t0, 1)
        log.info("모델을 올렸다 (%.1f초). 어댑터: %s", self.load_seconds, ", ".join(self.adapter_dirs))

    def describe(self) -> dict:
        return {"engine": "transformers", "quantized": bool(self.load_4bit),
                "base_model": self.base_model, "revision": self.revision, "device": self.device, "dtype": self.dtype,
                "four_bit": bool(self.load_4bit), "batch_size": self.batch_size,
                "adapters": list(self.adapter_dirs), "load_seconds": self.load_seconds}

    # ---- 추론 ---------------------------------------------------------------
    def _forward(self, prompts: list[str], n_options: int) -> list[list[float]]:
        import torch
        out: list[list[float]] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = self._tok(prompts[i:i + self.batch_size], return_tensors="pt", padding=True).to(self._model.device)
            # 왼쪽을 채운 줄도 실제 토큰의 위치 번호가 0부터 시작하게 한다. 넘기지 않으면 채운 만큼 밀려
            # 한 개씩 돌린 값과 아주 조금 달라진다.
            mask = batch["attention_mask"]
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                logits = self._model(input_ids=batch["input_ids"], attention_mask=mask,
                                     position_ids=positions).logits[:, -1].float()
            probs = torch.softmax(logits, -1)
            for row in probs:
                p = [row[a].item() + row[b].item() for a, b in self._letters[:n_options]]
                total = sum(p)
                if not total > 0:
                    raise LiveError("모델이 보기 글자에 확률을 주지 않았다")
                out.append([round(x / total, 5) for x in p])
        return out

    def _forward_with(self, adapter: str, prompts: list[str], n_options: int) -> list[list[float]]:
        if adapter == BASE:
            with self._model.disable_adapter():
                return self._forward(prompts, n_options)
        self._model.set_adapter(adapter)
        return self._forward(prompts, n_options)
