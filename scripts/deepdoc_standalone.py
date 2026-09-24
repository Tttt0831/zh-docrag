#!/usr/bin/env python
"""让 deepdoc 脱离 RAGFlow 整套服务单独运行。

问题：deepdoc/vision/ocr.py 和 deepdoc/parser/pdf_parser.py 都 `from common import settings`，
而 common/settings.py 在模块顶层 import 了 RAGFlow 的**整个存储层**——
es_conn、infinity_conn、ob_conn、opensearch_conn、gaussdb_conn、azure、gcs、minio、
opendal、redis、s3、oss，外加 rag.nlp.search。于是想跑个 OCR 就得装十几个数据库客户端。

但 deepdoc 实际只用到 settings 里的**一个变量**：PARALLEL_DEVICES（int，GPU 张数，
用来决定 OCR 并行度）。全部引用共 8 处，都是 `settings.PARALLEL_DEVICES`。

所以这里在 import deepdoc 之前，先往 sys.modules 里塞一个只含该变量的 stub，
把那条重依赖链整个断掉。这不改变 deepdoc 的任何行为——它读到的值和真 settings 一样。

用法：
    import deepdoc_standalone          # 必须在 import deepdoc 之前
    from deepdoc.vision import OCR
"""
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAGFLOW = REPO / "ragflow"

# deepdoc 用 get_project_base_directory() 定位 rag/res/deepdoc 下的 onnx 模型；
# 该函数优先读 RAG_PROJECT_BASE。模型软链在 ragflow/rag/res/deepdoc -> models/deepdoc。
os.environ.setdefault("RAG_PROJECT_BASE", str(RAGFLOW))
if str(RAGFLOW) not in sys.path:
    sys.path.insert(0, str(RAGFLOW))


def enable_cuda_libs() -> bool:
    """把 pip 装的 nvidia 库目录加进 LD_LIBRARY_PATH，让 onnxruntime 能加载 CUDA EP。

    onnxruntime.get_available_providers() 里有 CUDAExecutionProvider 只代表**编译时**
    支持；运行时若找不到 libcudnn.so.9 / libcublas，会**静默回落到 CPU**，不报错、
    不抛异常，只在 stderr 打一行 W 级日志。实测回落后 OCR 是 45s/页，差一个数量级。

    做法是用 ctypes 以 RTLD_GLOBAL 预加载这些 .so，让它们进入进程的全局符号表，
    之后 onnxruntime 再 dlopen libonnxruntime_providers_cuda.so 时依赖就已满足。
    torch 也是这么干的。

    （早先试过改 LD_LIBRARY_PATH + os.execv 重启进程，那是错的：脚本经 stdin 传入时
    重启后 stdin 已被消费，进程会静默什么都不做就退出。ctypes 预加载没有这个问题。）

    返回是否成功预加载。
    """
    import ctypes
    import glob

    # 用 sys.prefix 而不是 Path(sys.executable).resolve()：venv 的 bin/python 是指向
    # 系统解释器的符号链接，resolve() 会跳出 venv，找到的是系统目录、里面没有 nvidia 包。
    site = Path(sys.prefix) / "lib" / f"python3.{sys.version_info.minor}" / "site-packages" / "nvidia"
    if not site.exists():
        for p in sys.path:  # 兜底：直接在 sys.path 里找
            cand = Path(p) / "nvidia"
            if cand.is_dir():
                site = cand
                break
        else:
            return False

    # 顺序有讲究：被依赖的先加载。cublasLt 是 cublas 的依赖。
    order = ["cuda_runtime", "cuda_nvrtc", "cublas", "cudnn"]
    loaded = 0
    for pkg in order:
        for so in sorted(glob.glob(str(site / pkg / "lib" / "*.so*"))):
            if "cublasLt" in so:  # 先加载 cublasLt，再加载 cublas
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    loaded += 1
                except OSError:
                    pass
    for pkg in order:
        for so in sorted(glob.glob(str(site / pkg / "lib" / "*.so*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                pass  # 有些 .so 是符号链接或非库文件，加载失败不致命
    return loaded > 0


def _device_count() -> int:
    """数 GPU。优先 torch（真 settings 就是这么数的），没有 torch 就问 nvidia-smi。"""
    override = os.environ.get("DEEPDOC_PARALLEL_DEVICES")
    if override is not None:
        return int(override)
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return len([ln for ln in out.stdout.splitlines() if ln.startswith("GPU ")])
    except Exception:
        return 0


def install(parallel_devices: int | None = None) -> int:
    """装上 stub。返回生效的 PARALLEL_DEVICES。重复调用安全。"""
    if isinstance(sys.modules.get("common.settings"), types.ModuleType) and getattr(
        sys.modules["common.settings"], "_deepdoc_stub", False
    ):
        return sys.modules["common.settings"].PARALLEL_DEVICES

    n = _device_count() if parallel_devices is None else parallel_devices

    import common  # 轻量：common/__init__.py 只有 15 行，不 import 任何重东西

    stub = types.ModuleType("common.settings")
    stub.PARALLEL_DEVICES = n

    # 照抄 common/settings.py:169-170 的原始逻辑，不是随便填的默认值。
    # DOC_ENGINE_INFINITY 为真时 rag_tokenizer.tokenize() 直接返回原文不分词
    # （Infinity 在服务端自己分词）；为假才走真正的中文分词器。
    # 我们不连任何 doc engine，必须走真分词，否则表格结构识别的 blockType()
    # 会拿到未分词文本，块类型判断全错。
    stub.DOC_ENGINE = os.getenv("DOC_ENGINE", "elasticsearch")
    stub.DOC_ENGINE_INFINITY = stub.DOC_ENGINE.lower() == "infinity"

    stub._deepdoc_stub = True
    sys.modules["common.settings"] = stub
    common.settings = stub
    return n


def stub_torch_if_absent(device_count: int) -> str:
    """DeepDoc 判断能否用 GPU 的唯一依据是 `import torch` 能否成功（ocr.py:89-93）：

        def cuda_is_available():
            pip_install_torch(); import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > target_id: return True

    之后它直接用 onnxruntime 建 CUDAExecutionProvider 会话，**torch 再无其它用途**
    （grep 过整个 deepdoc/，只有这一处）。为一个布尔值装 4 GB 的 torch + nvidia 依赖
    不值得，所以这里在真 torch 缺席时注入一个最小 stub。

    另有一个务必知道的副作用：pip_install_torch() 在 DEVICE != "cpu" 时会**运行时**执行
    `pip install "torch>=2.5.0,<3.0.0"` —— 没有 CUDA 构建上界，会拉到 cu13x 的 wheel，
    而本机驱动上限 CUDA 12.4，装得上跑不了。注入 stub 后这条路径根本不会被触发。

    装了真 torch 的环境不受影响：本函数检测到真 torch 就直接返回。
    设 DEEPDOC_NO_TORCH_STUB=1 可禁用注入。
    """
    if os.environ.get("DEEPDOC_NO_TORCH_STUB") == "1":
        return "disabled"
    try:
        import torch  # noqa: F401
        return "real"
    except ImportError:
        pass

    torch_mod = types.ModuleType("torch")
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: device_count > 0
    cuda_mod.device_count = lambda: device_count
    torch_mod.cuda = cuda_mod

    # scipy 的 array_api_compat 一见 sys.modules 里有 torch，就会对每个数组做
    # issubclass(type(x), torch.Tensor) 来判断类型。stub 里没有 Tensor 会直接
    # AttributeError 崩在 scipy 内部，报错位置离真正原因很远，很难查。
    # 给个空类即可：numpy 数组不是它的子类，判断自然返回 False。
    class _Tensor:
        pass

    torch_mod.Tensor = _Tensor
    torch_mod.__version__ = "0.0.0+deepdoc-stub"
    torch_mod._deepdoc_stub = True
    sys.modules["torch"] = torch_mod
    sys.modules["torch.cuda"] = cuda_mod
    return "stub"


CUDA_LIBS_OK = enable_cuda_libs()
PARALLEL_DEVICES = install()
# 只在 CUDA 库确实加载成功时才谎称有 GPU；否则让 deepdoc 老实走 CPU，
# 免得它建 CUDA 会话失败后以难懂的方式崩掉。
TORCH_MODE = stub_torch_if_absent(PARALLEL_DEVICES if CUDA_LIBS_OK else 0)
