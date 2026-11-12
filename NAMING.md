# Venturi 命名规范

本规范基于仓库实际演进中形成的约定整理,约束新增代码的命名选择。

## 核心原则

**架构名只出现在适配层,机制层永远中立。**

```
✅ weight_source.cuh    ← 机制:读权重字节(任何模型都用)
✅ staged_uploader.cuh  ← 机制:pinned 中转上传
✅ gemm_dispatch.cuh    ← 机制:GEMM 派发
✅ codecs.py            ← 机制:格式注册表

✅ qwen_runtime.cuh     ← 适配:Qwen 架构的运行时
✅ qwen_load/           ← 适配:Qwen 加载管线
✅ flash_next_*.py      ← 适配:Flash-Next 模型
```

## 分层规则

### 1. 机制层(C++ / ops/csrc/)

| 项 | 规则 | 示例 |
|---|---|---|
| 文件名 | 小写 + 下划线,描述机制而非用途 | `weight_source.cuh`, `staged_uploader.cuh` |
| 命名空间 | `namespace vr`(由外层提供,不自包装) | — |
| 结构体 | `VR` 前缀 + PascalCase | `VRConfig`, `VRMatrix`, `VRGemm` |
| 结构体字段 | snake_case,简洁 | `weight_bits`, `dtype`, `shadow` |
| 函数 | snake_case | `gemm_project()`, `read_bytes()`, `upload()` |
| C 导出 | `venturi_` 前缀 | `venturi_q4_fused_run()`, `venturi_dense_run()` |
| 内核 | `venturi_{codec}_{op}_run` 模式 | `venturi_q4_fused_run`, `venturi_row_dequant_rows` |

### 2. 机制层(Python / venturi/ 顶层)

| 项 | 规则 | 示例 |
|---|---|---|
| 模块名 | 小写,无架构名 | `codecs.py`, `storage_verify.py`, `ninfer.py` |
| 常量 | UPPER_CASE | `CODECS`, `EXECUTION_QUALIFIED`, `MAGIC` |
| dataclass | PascalCase | `Codec`, `LoadSpec`, `Budget` |
| 函数 | snake_case | `codec_for()`, `dequant_row_reference()` |
| 私有 | `_` 前缀 | `_bind()`, `_digest()` |

### 3. 架构适配层(Python / models/)

| 项 | 规则 | 示例 |
|---|---|---|
| 模块/包 | 架构名前缀 | `qwen_native.py`, `qwen_load/`, `flash_next.py` |
| 类名 | PascalCase,含架构名 | `NativeQwenModel`, `ResidentQwenModel` |
| 加载管线包 | `{arch}_load/` | `qwen_load/` |

### 4. 架构适配层(C++ / ops/csrc/)

| 项 | 规则 | 示例 |
|---|---|---|
| 文件名 | `{arch}_` 前缀 | `qwen_runtime.cuh`, `qwen_head_gdn.cuh` |
| 导出 | `venturi_{arch}_` 前缀 | `venturi_qwen_create()`, `venturi_qwen_forward()` |

### 5. 测试

| 项 | 规则 | 示例 |
|---|---|---|
| 文件 | `test_{主题}.py` | `test_row_codec_execution.py`, `test_codec_loading.py` |
| 测试函数 | `test_{行为}_{条件}` | `test_dequant_resident_matches_ondemand` |
| 参数化 | `[param]` 后缀,描述而非编号 | `[native-fp16-q4_g64_fp16]` |

### 6. 分支 / 提交

| 项 | 规则 | 示例 |
|---|---|---|
| 分支 | kebab-case,描述主题 | `codec-generalization` |
| Commit | Conventional Commits | `feat:`, `fix:`, `test:`, `refactor:`, `ci:` |
| PR 标题 | `{type}: {描述}` | `feat: codec-generalized loading...` |

## Codec 命名

格式标识符遵循 `{bits}g{group}_{scale_dtype}` 模式:

```
q8_g32_fp16    ← 8-bit, group 32, FP16 scales
q4_g64_fp16    ← 4-bit, group 64, FP16 scales
q6_g64_fp16    ← 6-bit, group 64, FP16 scales
fp16           ← dense FP16(无量化参数)
bf16 / fp32    ← dense 浮点
```

执行资格表键为 GPU compute capability 元组:

```python
EXECUTION_QUALIFIED = {
    (7, 0): {"q8_g32_fp16"},          # V100 (Volta)
    (8, 9): {"q8_g32_fp16", "q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16"},  # L40/4080 (Ada)
}
```

## 禁止

- ❌ 机制层文件/函数/变量带架构名(`qwen_source.cuh` → 已改名 `weight_source.cuh`)
- ❌ 适配层引用其他适配层的内部实现(跨架构走机制层)
- ❌ 混合命名风格(同一层内保持一致)
- ❌ 缩写不含上下文(`ws` 不如 `weight_source` 清晰)
