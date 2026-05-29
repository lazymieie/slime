import argparse
import inspect
import json
from pathlib import Path

from megatron.bridge import AutoBridge
import slime_plugins.megatron_bridge  # 注册 slime bridge/plugin
from slime.utils.megatron_bridge_utils import patch_auto_bridge_hf_config


def collect_hf_tensor_names(model_path: str):
    model_path = Path(model_path)
    names = set()

    # safetensors index
    for index_name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        index_file = model_path / index_name
        if index_file.exists():
            with open(index_file, "r") as f:
                data = json.load(f)
            names.update(data.get("weight_map", {}).keys())

    # single/multiple safetensors files
    if not names:
        try:
            from safetensors import safe_open
            for f in model_path.glob("*.safetensors"):
                with safe_open(str(f), framework="pt", device="cpu") as sf:
                    names.update(sf.keys())
        except Exception as e:
            print("[WARN] Cannot read safetensors directly:", repr(e))

    return names


def expected_hf_names_for_mtp_megatron_param(name: str):
    """
    根据 slime test_qwen3_5_mtp_bridge_mapping.py 里的规则写的最小手动映射检查。
    只用于验证 HF checkpoint 中是否存在对应权重。
    """
    # 去掉 Megatron VLM 外层前缀
    if name.startswith("language_model."):
        name = name[len("language_model."):]

    # examples:
    # mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight57
    # mtp.layers.0.transformer_layer.mlp.shared_experts.linear_fc2.weight
    parts = name.split(".")
    if not name.startswith("mtp.layers.0.transformer_layer.mlp."):
        return []

    prefix = "mtp.layers.0.mlp."

    if "experts.linear_fc1.weight" in name:
        expert_id = name.split("experts.linear_fc1.weight", 1)[1]
        return [
            f"{prefix}experts.{expert_id}.gate_proj.weight",
            f"{prefix}experts.{expert_id}.up_proj.weight",
        ]

    if "experts.linear_fc2.weight" in name:
        expert_id = name.split("experts.linear_fc2.weight", 1)[1]
        return [
            f"{prefix}experts.{expert_id}.down_proj.weight",
        ]

    if "shared_experts.linear_fc1.weight" in name:
        return [
            f"{prefix}shared_expert.gate_proj.weight",
            f"{prefix}shared_expert.up_proj.weight",
        ]

    if "shared_experts.linear_fc2.weight" in name:
        return [
            f"{prefix}shared_expert.down_proj.weight",
        ]

    if "shared_experts.gate_weight" in name:
        return [
            f"{prefix}shared_expert_gate.weight",
        ]

    if "mlp.linear_fc1.weight" in name:
        return [
            f"{prefix}gate_proj.weight",
            f"{prefix}up_proj.weight",
        ]

    if "mlp.linear_fc2.weight" in name:
        return [
            f"{prefix}down_proj.weight",
        ]

    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/gemini/space/gjx/slime_info/Qwen3.5-35B-A3B",
        help="HF checkpoint directory",
    )
    args = parser.parse_args()

    print("=== Build AutoBridge ===")
    bridge = patch_auto_bridge_hf_config(
        AutoBridge.from_hf_pretrained(args.model_path, trust_remote_code=True)
    )

    print("AutoBridge:", type(bridge))
    real_bridge = getattr(bridge, "_model_bridge", None)
    print("Real bridge:", type(real_bridge))
    if real_bridge is not None:
        print("Real bridge file:", inspect.getfile(type(real_bridge)))
        print("Has _convert_mtp_param:", hasattr(real_bridge, "_convert_mtp_param"))

    print("\n=== Provider MTP fields ===")
    provider = bridge.to_megatron_provider(load_weights=False)
    print("Provider:", type(provider))
    for k in dir(provider):
        if "mtp" in k.lower() or "next" in k.lower() or "predict" in k.lower():
            try:
                print(f"{k}: {getattr(provider, k)}")
            except Exception as e:
                print(f"{k}: <ERR {e}>")

    print("\n=== HF checkpoint MTP tensors ===")
    hf_names = collect_hf_tensor_names(args.model_path)
    mtp_names = sorted([n for n in hf_names if "mtp" in n.lower()])
    print("Total HF tensors:", len(hf_names))
    print("Total HF MTP tensors:", len(mtp_names))
    for n in mtp_names[:50]:
        print("HF_MTP:", n)
    if len(mtp_names) > 50:
        print(f"... {len(mtp_names) - 50} more")

    print("\n=== Check expected mapping for sample Megatron MTP params ===")
    sample_megatron_params = [
        "language_model.mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight0",
        "language_model.mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight57",
        "language_model.mtp.layers.0.transformer_layer.mlp.shared_experts.linear_fc1.weight",
        "language_model.mtp.layers.0.transformer_layer.mlp.shared_experts.linear_fc2.weight",
        "language_model.mtp.layers.0.transformer_layer.mlp.shared_experts.gate_weight",
    ]

    ok_all = True
    for mp in sample_megatron_params:
        expected = expected_hf_names_for_mtp_megatron_param(mp)
        print(f"\nMEGATRON: {mp}")
        print("EXPECTED HF:", expected)
        for h in expected:
            exists = h in hf_names
            ok_all = ok_all and exists
            print(" ", "OK " if exists else "MISS", h)

    print("\n=== Result ===")
    if ok_all:
        print("HF checkpoint contains expected MTP MoE tensors.")
        print("如果训练仍然 No mapping found，说明当前 megatron.bridge 的 Qwen35VLMoEBridge 没有实际注册/使用这些映射逻辑。")
    else:
        print("Some expected HF MTP tensors are missing. 需要检查 checkpoint 结构或映射规则。")


if __name__ == "__main__":
    main()
