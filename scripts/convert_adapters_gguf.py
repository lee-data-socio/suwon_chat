"""LoRA 어댑터(PEFT, safetensors)를 llama.cpp가 읽는 형식(GGUF)으로 바꾼다.

    python scripts/convert_adapters_gguf.py        # adapters/exp*/adapter_model.safetensors → adapters/exp*/adapter.gguf

GPU 없이 양자화 모델로 돌릴 때(PREDICT_BACKEND=gguf) 쓴다. 바꾼 파일은 저장소에 들어 있으므로 다시 돌릴 일은
어댑터를 새로 학습했을 때뿐이다. `pip install gguf`가 필요하다.

형식: 기본 모델의 텐서 이름 뒤에 `.lora_a`, `.lora_b`를 붙인 텐서 두 개씩(float32).
배율(alpha / r)은 llama.cpp가 `adapter.lora.alpha`와 텐서 모양에서 계산한다.

주의할 점 하나: llama.cpp는 Llama 계열의 q_proj·k_proj 가중치를 행 순서를 바꿔서 저장한다(허깅페이스와 회전
위치 임베딩의 짝짓는 방식이 달라서다. llama.cpp `convert_hf_to_gguf.py`의 `LlamaModel.permute`). 그래서 어댑터의
q_proj·k_proj 쪽 B 행렬도 같은 순서로 바꿔야 한다. 바꾸지 않으면 q 쪽 어댑터가 엉뚱한 행에 더해져, 오류 없이
돌지만 예측이 저장된 값과 크게 달라진다(2026-11-15에 실제로 겪었다. 거리 0.19 → 고친 뒤는 README 참고).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output"}
# 기본 모델(Bllossom 8B = Llama-3 8B)의 주의 머리 수. 허깅페이스 config.json의 num_attention_heads, num_key_value_heads
N_HEAD, N_HEAD_KV = 32, 8


def permute_rows(weights, n_head: int):
    """llama.cpp `LlamaModel.permute`와 같은 행 순서 바꿈. (행, 열) 모양의 B 행렬에 쓴다."""
    rows = weights.shape[0]
    return (weights.reshape(n_head, 2, rows // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))
NAME = re.compile(r"^base_model\.model\.model\.layers\.(\d+)\.self_attn\.(\w+)\.lora_([AB])\.weight$")


def convert(folder: Path) -> Path:
    import gguf
    import numpy as np
    from safetensors import safe_open

    cfg = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("bias", "none") != "none":
        raise SystemExit(f"{folder.name}: 이 변환기는 기본 LoRA만 다룬다 (DoRA, rsLoRA, bias 학습은 안 된다)")
    out = folder / "adapter.gguf"
    writer = gguf.GGUFWriter(str(out), arch="llama")
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(cfg["lora_alpha"]))
    writer.add_name(f"suwon-survey-{folder.name}")
    n = 0
    with safe_open(str(folder / "adapter_model.safetensors"), "np") as f:
        for key in sorted(f.keys()):
            m = NAME.match(key)
            if not m or m.group(2) not in MODULES:
                raise SystemExit(f"{folder.name}: 모르는 텐서 이름 {key}")
            layer, module, ab = m.groups()
            tensor = np.ascontiguousarray(f.get_tensor(key), dtype=np.float32)
            rank = cfg["r"]
            if (ab == "A" and tensor.shape[0] != rank) or (ab == "B" and tensor.shape[1] != rank):
                raise SystemExit(f"{folder.name}: {key}의 모양 {tensor.shape}이 r={rank}와 맞지 않는다")
            if ab == "B" and module in ("q_proj", "k_proj"):       # 기본 모델의 q·k와 같은 행 순서로
                tensor = np.ascontiguousarray(permute_rows(tensor, N_HEAD if module == "q_proj" else N_HEAD_KV))
            writer.add_tensor(f"blk.{layer}.{MODULES[module]}.weight.lora_{ab.lower()}", tensor)
            n += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"{folder.name}: 텐서 {n}개 → {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    folders = [Path(p) for p in sys.argv[1:]] or sorted(p for p in (ROOT / "adapters").iterdir() if (p / "adapter_config.json").is_file())
    for folder in folders:
        convert(folder)
