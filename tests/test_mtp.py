
# demo_qwen3_5_mtp.py
import torch
from megatron.bridge.models.qwen_v1.qwen35_vl_bridge import Qwen35VLMoEBridge
from slime_plugins.megatron_bridge_utils import patch_auto_bridge_hf_config
from megatron.bridge.models.conversion.auto_bridge import AutoBridge

# HF checkpoint 路径
hf_checkpoint_path = "/gemini/space/gjx/slime_info/Qwen3.5-4B"

# 1. 使用 AutoBridge 加载 HF 配置（不加载权重）
bridge = patch_auto_bridge_hf_config(
    AutoBridge.from_hf_pretrained(hf_checkpoint_path, trust_remote_code=True)
)

print("bridge type:", type(bridge))

# 2. 获取 provider（Megatron 对象）
provider = bridge.to_megatron_provider(load_weights=False)
print("provider type:", type(provider))

# 3. 打印 provider 内部 MTP 相关字段
print("\nProvider MTP fields:")
for k in dir(provider):
    if "mtp" in k.lower() or "predict" in k.lower() or "next" in k.lower():
        print(f"  {k}: {getattr(provider, k)}")

# 4. 也可以直接实例化 Megatron-Bridge 模型，查看参数
model = Qwen35VLMoEBridge()
print("\nModel parameters containing 'mtp':")
for name, param in model.named_parameters():
    if "mtp" in name.lower():
        print(f"{name} | shape: {param.shape} | dtype: {param.dtype}")