import platform
import torch
import subprocess
import pkg_resources

print("=" * 60)
print("🔍 Python / CUDA / PyTorch 环境检测脚本")
print("=" * 60)

# --- Python 环境 ---
print(f"🐍 Python 版本: {platform.python_version()}")
print(f"🧠 系统平台: {platform.system()} {platform.release()}")
print()

# --- PyTorch 信息 ---
print("🔥 PyTorch 信息:")
print(f"  torch 版本: {torch.__version__}")
print(f"  CUDA runtime: {torch.version.cuda}")
print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
print(f"  CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 数量: {torch.cuda.device_count()}")
print()

# --- 驱动信息 (通过 nvidia-smi) ---
try:
    smi_output = subprocess.check_output(["nvidia-smi"], text=True)
    print("🧩 nvidia-smi 输出:")
    print(smi_output.split("\n")[0])
    for line in smi_output.split("\n"):
        if "Driver Version" in line:
            print(line.strip())
            break
except Exception as e:
    print("⚠️ 无法运行 nvidia-smi:", e)
print()

# --- 检查主要 CUDA 相关包 ---
print("📦 CUDA 相关 Python 包:")
cuda_packages = ["torch", "torchvision", "torchaudio", "triton", "xformers", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12"]
for dist in pkg_resources.working_set:
    if any(name in dist.project_name.lower() for name in cuda_packages):
        print(f"  {dist.project_name}=={dist.version}")
print("=" * 60)
