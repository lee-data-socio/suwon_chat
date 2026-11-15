"""실시간 백엔드를 큰 모델 없이 돌려 보기 위한 아주 작은 Llama와 어댑터 세 개를 만든다.

가중치가 무작위라 예측은 뜻이 없다. 모델 올리기, 어댑터 갈아 끼우기, 확률 계산의 배선만 확인한다.

    python tests/tiny.py <폴더>      # <폴더>/base, <폴더>/adapters/exp1~3 을 만든다
    LIVE_BASE_MODEL=<폴더>/base SUWON_ADAPTER_DIR=<폴더>/adapters LIVE_DTYPE=float32 PREDICT_BACKEND=live \\
        python -m uvicorn suwon_chat.api:app --port 8010
"""
import sys
from pathlib import Path


def make_tiny(root: Path, texts: list[str]) -> tuple[Path, dict[str, Path]]:
    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    # 글자 하나가 토큰 하나인 토크나이저
    chars = sorted(set("".join(texts)) | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ Answer:."))
    vocab = {t: i for i, t in enumerate(["<pad>", "<s>", "</s>", "<unk>"] + chars)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                  pad_token="<pad>")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2)
    base = root / "base"
    LlamaForCausalLM(cfg).save_pretrained(base)
    tok.save_pretrained(base)

    adapters = {}
    for seed, name in enumerate(["exp1", "exp2", "exp3"], start=1):
        torch.manual_seed(seed)
        model = get_peft_model(LlamaForCausalLM.from_pretrained(base),
                               LoraConfig(r=4, lora_alpha=64, target_modules=["q_proj", "v_proj"],
                                          init_lora_weights=False, task_type="CAUSAL_LM"))
        for n, p in model.named_parameters():      # 어댑터마다 뚜렷이 다른 값을 준다
            if "lora_" in n:
                torch.nn.init.normal_(p, std=0.5)
        path = root / "adapters" / name
        model.save_pretrained(path)
        adapters[name] = path
    return base, adapters


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from suwon_chat.store import Store
    store = Store()
    texts = [q.text + "".join(q.all_options) for q in store.questions.values()]
    texts += list(store.steering.values()) + [g for gs in store.groups.values() for g in gs]
    texts.append("Answer the following question keeping in mind your previous answers. Question as a choice between,\n")
    base, adapters = make_tiny(Path(sys.argv[1]), texts)
    print("기본 모델:", base)
    print("어댑터:", {k: str(v) for k, v in adapters.items()})
