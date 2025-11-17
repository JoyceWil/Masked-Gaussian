import sys
import os
import platform
import subprocess
import importlib


def print_header(title):
    """打印一个漂亮的标题"""
    print("\n" + "=" * 80)
    print(f"--- {title} ---")
    print("=" * 80)


def run_command(command, shell=True):
    """运行一个 shell 命令并返回其输出，如果出错则返回错误信息"""
    try:
        # 使用 'utf-8' 编码，如果不行则回退到 'latin-1'
        result = subprocess.run(command, shell=shell, check=True, capture_output=True, text=True, encoding='utf-8')
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # 即使命令失败，stderr 也可能包含有用的信息
        return f"Error running command: {' '.join(command)}\nStdout: {e.stdout.strip()}\nStderr: {e.stderr.strip()}"
    except UnicodeDecodeError:
        # 捕获编码错误并重试
        try:
            result = subprocess.run(command, shell=shell, check=True, capture_output=True, text=True,
                                    encoding='latin-1')
            return result.stdout.strip()
        except Exception as e:
            return f"Error running command with fallback encoding: {e}"
    except FileNotFoundError:
        return f"Error: Command '{command[0] if isinstance(command, list) else command.split()[0]}' not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def check_python():
    """1. 检查 Python 环境"""
    print_header("1. Python & OS Environment")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    if platform.architecture()[0] != '64bit':
        print("\n!!! 警告: 你似乎在运行 32 位 Python。这与 64 位 CUDA 库不兼容 !!!\n")
    print(f"Conda Environment: {os.environ.get('CONDA_DEFAULT_ENV', '未检测到 (可能不是 Conda 或未激活)')}")


def check_pytorch():
    """2. 检查 PyTorch 安装"""
    print_header("2. PyTorch & CUDA (Python 视角)")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")

        is_cuda = torch.cuda.is_available()
        print(f"PyTorch: CUDA 是否可用? {is_cuda}")

        if is_cuda:
            print(f"PyTorch: 侦测到的 GPU 数量: {torch.cuda.device_count()}")
            print(f"PyTorch: 当前 GPU 名称: {torch.cuda.get_device_name(0)}")
            print(f"PyTorch: 编译时所用的 CUDA 版本: {torch.version.cuda}")
        else:
            print("\n!!! 错误: PyTorch 报告 CUDA 不可用。这很可能是问题的根源 !!!")
            print("请检查你的 PyTorch 是否为 CPU 版本 (例如 '2.1.0+cpu')。")
            print("如果是，请卸载并重新安装 GPU 版本的 PyTorch。")

    except ImportError:
        print("\n!!! 错误: PyTorch 未安装。请在 (r2_gaussian) 环境中安装 PyTorch。")
    except Exception as e:
        print(f"\n导入 PyTorch 时发生意外错误: {e}")


def check_system_tools():
    """3. 检查系统编译工具 (CUDA Toolkit 和 C++ 编译器)"""
    print_header("3. 系统编译工具 (外部工具)")

    # 检查 nvcc (CUDA Toolkit)
    print("\n--- 检查 CUDA Toolkit (nvcc) ---")
    nvcc_output = run_command(['nvcc', '--version'])
    if "Error" in nvcc_output or "not found" in nvcc_output:
        print("!!! 警告: 'nvcc' (CUDA Toolkit) 未在系统 PATH 中找到。")
        print("    这对于 *编译* C++/CUDA 扩展是必需的。")
    else:
        # 尝试从输出中提取版本
        version_line = [line for line in nvcc_output.splitlines() if "Cuda compilation tools" in line]
        if version_line:
            print(f"NVCC (Toolkit) 版本: {version_line[0]}")
        else:
            print(f"NVCC 输出: \n{nvcc_output}")

    # 检查 cl.exe (MSVC C++ Compiler)
    print("\n--- 检查 C++ 编译器 (cl.exe) ---")
    # cl.exe 在没有参数时会返回错误，但这证明它存在
    cl_output = run_command(['cl.exe'])
    if "not found" in cl_output or "Error" in cl_output and "Microsoft" not in cl_output:
        print("!!! 警告: 'cl.exe' (Visual Studio C++ 编译器) 未在系统 PATH 中找到。")
        print("    这对于 *编译* C++/CUDA 扩展是必需的。")
        print("    请尝试从 'Developer Command Prompt for VS' 激活你的 Conda 环境并运行此脚本。")
    else:
        # 尝试从 stderr 抓取版本信息 (cl.exe 把版本信息输出到 stderr)
        try:
            result = subprocess.run(['cl.exe'], shell=True, capture_output=True, text=True, encoding='latin-1',
                                    timeout=3)
            version_line = result.stderr.splitlines()[0]
            print(f"CL.exe (编译器) 版本: {version_line}")
        except Exception:
            print("CL.exe 存在，但获取版本失败。通常这没问题。")


def check_nvidia_driver():
    """4. 检查 NVIDIA 驱动 (nvidia-smi)"""
    print_header("4. NVIDIA 驱动 (nvidia-smi 视角)")
    smi_output = run_command(['nvidia-smi'])
    if "Error" in smi_output or "not found" in smi_output:
        print("!!! 错误: 'nvidia-smi' 未找到。你的 NVIDIA 驱动是否正确安装?")
    else:
        # 提取驱动报告的 CUDA 版本
        cuda_line = [line for line in smi_output.splitlines() if "CUDA Version:" in line]
        if cuda_line:
            driver_cuda_version = cuda_line[0].split('CUDA Version:')[1].strip().split(' ')[0]
            print(f"驱动程序支持的最高 CUDA 版本: {driver_cuda_version}")
            print("注意: 这 *不是* 你安装的 Toolkit 版本，而是驱动的兼容上限。")
            print("PyTorch 的 CUDA 版本 (见第 2 节) 必须 *小于或等于* 这个版本。")
        else:
            print(f"NVIDIA-SMI 输出 (未找到 CUDA 版本行): \n{smi_output}")


def check_problematic_import():
    """5. 尝试复现导入错误"""
    print_header("5. 诊断 'xray_gaussian_rasterization_voxelization'")

    lib_name = "xray_gaussian_rasterization_voxelization"

    print(f"--- 步骤 5.1: 检查 '{lib_name}' 是否已安装...")
    try:
        spec = importlib.util.find_spec(lib_name)
        if spec:
            print(f"成功: '{lib_name}' 已安装。位置: {spec.origin}")
        else:
            print(f"!!! 错误: '{lib_name}' 未安装在此环境中。")
            print("你是否运行了 'pip install .' (在子目录中)？")
            return  # 无法继续
    except Exception as e:
        print(f"查找库时出错: {e}")
        return

    print(f"\n--- 步骤 5.2: 尝试导入 '{lib_name}' (顶层)...")
    try:
        import xray_gaussian_rasterization_voxelization
        print("成功: 顶层导入工作正常。")
    except Exception as e:
        print(f"!!! 错误: 顶层导入失败: {e}")
        print("这表明库的 `__init__.py` 文件有语法错误。")
        return

    print(f"\n--- 步骤 5.3: 尝试导入 '... import _C' (复现错误)...")
    print("这是你运行 'train.py' 时失败的步骤。")
    try:
        from xray_gaussian_rasterization_voxelization import _C
        print("\n**********************************************************")
        print("!!! 成功: 成功导入 'from . import _C' !!!")
        print("**********************************************************")
        print("这很奇怪。如果此脚本成功，但 'train.py' 失败，")
        print("请确保你是从 *完全相同的终端* 运行 'train.py' 的。")
    except ImportError as e:
        print("\n**********************************************************")
        print("!!! 失败 (已预料): 复现了导入错误。")
        print(f"错误信息: {e}")
        print("**********************************************************")
        print("\n--- 诊断结论 ---")
        print("这 99% 确认是一个环境不匹配问题。请检查：")
        print("1. [第 2 节] PyTorch 是否为 'CPU' 版本？(如果是，这是问题所在)")
        print("2. [第 2 节] PyTorch 的 CUDA 版本 (例如 11.8) 是否与")
        print("   [第 3 节] NVCC (Toolkit) 的版本 (例如 11.8) 匹配？")
        print("3. [第 2 节] PyTorch 的 CUDA 版本 (例如 11.8) 是否 *低于*")
        print("   [第 4 节] 驱动的 CUDA 版本 (例如 12.2)？")
        print("4. [第 3 节] 'cl.exe' 和 'nvcc' 是否都已找到？(如果未找到，编译会失败)")

    except Exception as e:
        print(f"\n发生意外错误: {e}")


if __name__ == "__main__":
    print("开始运行 R2-Gaussian 环境诊断脚本...")
    check_python()
    check_pytorch()
    check_system_tools()
    check_nvidia_driver()
    check_problematic_import()
    print("\n" + "=" * 80)
    print("诊断完成。")
    print("=" * 80)