"""测试模型加载"""
import torch
import argparse

# 检查checkpoint内容
ckpt_path = './output_cwd_class/checkpoint-last.pth'
print(f"Loading checkpoint from: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location='cpu')
print(f"\nCheckpoint keys: {list(ckpt.keys())}")

if 'model' in ckpt:
    print(f"\nModel state dict keys (first 10):")
    keys = list(ckpt['model'].keys())[:10]
    for k in keys:
        print(f"  {k}")
    print(f"  ... total {len(ckpt['model'])} keys")

if 'args' in ckpt:
    print(f"\nSaved args: {ckpt['args']}")

if 'epoch' in ckpt:
    print(f"\nEpoch: {ckpt['epoch']}")

# 尝试加载模型
print("\n" + "="*50)
print("Testing model loading...")
print("="*50)

from denoiser_cwd_class import CWDDenoiserWithClass

args = argparse.Namespace(
    model='JiT-CWD-Class-Base',
    img_size=224,
    in_channels=1,
    num_classes=16,
    class_dropout_prob=0.1,
    attn_dropout=0.0,
    proj_dropout=0.0,
    P_mean=-0.8,
    P_std=0.8,
    noise_scale=1.0,
    t_eps=0.05,
    sampling_method='heun',
    num_sampling_steps=50,
)

model = CWDDenoiserWithClass(args)
print(f"Model created successfully")

# 加载权重
if 'model' in ckpt:
    state_dict = ckpt['model']
    # 检查是否需要去掉前缀
    model_state = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            model_state[k[6:]] = v
        else:
            model_state[k] = v
    
    # 加载
    missing, unexpected = model.model.load_state_dict(model_state, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  Missing: {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected: {unexpected[:5]}...")

print("\nModel loaded successfully!")

# 测试推理
print("\n" + "="*50)
print("Testing inference...")
print("="*50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = model.to(device)
model.eval()

# 创建测试输入
test_input = torch.randn(1, 1, 224, 224).to(device)
test_label = torch.tensor([0]).to(device)  # BPSK

with torch.no_grad():
    output = model.denoise(test_input, test_label, cfg_scale=2.0)

print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

print("\n✓ All tests passed!")
