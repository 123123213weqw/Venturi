"""Load pipeline for NativeQwenModel.from_q8, split by responsibility.

spec         argument/profile validation and device plan (LoadSpec)
planner      source re-open and config geometry checks
budget       residency byte accounting and memory admission (Budget)
descriptors  constant uploads and ctypes ABI assembly (Descriptors)
verify       projection info and post-create native reconciliation
abi          ctypes descriptor structs and native symbol binding

qwen_native.from_q8 orchestrates these stages in the historical order; each
stage raises the historical messages so behavior is unchanged.
"""
