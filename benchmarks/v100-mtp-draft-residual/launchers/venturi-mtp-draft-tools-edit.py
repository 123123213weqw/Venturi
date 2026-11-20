from pathlib import Path
root=Path.cwd()
for name in ['qualify_qwen_mtp','qualify_qwen_mtp_batch','profile_qwen_mtp','profile_qwen_mtp_batch','serve_native_benchmark']:
 p=root/'tools'/(name+'.py');s=p.read_text();key='    a = p.parse_args()' if name!='serve_native_benchmark' else '    args = p.parse_args()'
 s=s.replace(key,'    p.add_argument("--mtp-draft-projection-profile", choices=("fp16", "q8-residual"), default="fp16")\n'+key,1)
 if name=='serve_native_benchmark':
  s=s.replace('    if args.target_layer_graph and (','    if args.mtp_draft_projection_profile != "fp16" and (not args.mtp_artifact or args.mode != "tp"):\n        p.error("Residual MTP draft requires an explicit draft artifact and non-PP target")\n    if args.target_layer_graph and (',1)
  s=s.replace('target, QwenMTPWeights.from_ninfer(NInferArtifact(args.mtp_artifact))','target, QwenMTPWeights.from_ninfer(NInferArtifact(args.mtp_artifact)),\n                draft_projection_profile=args.mtp_draft_projection_profile',1)
  s=s.replace('mtp_draft_projection_profile="fp16" if args.mtp_artifact else None,','mtp_draft_projection_profile=model.projection_info["mtp_draft_projection_profile"] if args.mtp_artifact else None,')
 elif name=='qualify_qwen_mtp':
  s=s.replace('mtp_draft_projection_profile="fp16",','mtp_draft_projection_profile=a.mtp_draft_projection_profile,')
  s=s.replace('model = NativeQwenMTPModel.from_q8(target, mtp_weights)','model = NativeQwenMTPModel.from_q8(target, mtp_weights, draft_projection_profile=a.mtp_draft_projection_profile)')
  s=s.replace('    if a.mode == "pp" and a.projection_profile != "fp16":','    if a.mtp_draft_projection_profile != "fp16" and (a.mode == "pp" or a.skip_block_reference):\n        p.error("New draft arithmetic requires non-PP and fresh block reference checks")\n    if a.mode == "pp" and a.projection_profile != "fp16":')
 else:
  s=s.replace('target, QwenMTPWeights.from_ninfer(NInferArtifact(a.mtp_artifact))','target, QwenMTPWeights.from_ninfer(NInferArtifact(a.mtp_artifact)),\n            draft_projection_profile=a.mtp_draft_projection_profile',1)
  s=s.replace('draft_profile="fp16",','draft_profile=a.mtp_draft_projection_profile,')
  s=s.replace('draft="fp16", kv="fp16"','draft=a.mtp_draft_projection_profile, kv="fp16"')
 p.write_text(s)
p=root/'src/venturi/qwen_native_cli.py';s=p.read_text();s=s.replace('    p.add_argument("--mtp-recovery",','    p.add_argument("--mtp-draft-projection-profile", choices=("fp16", "q8-residual"), default="fp16")\n    p.add_argument("--mtp-recovery",',1)
s=s.replace('        if args.mtp_recovery != "replay" and (','        if args.mtp_draft_projection_profile != "fp16" and (not args.mtp_artifact or args.dtype != "fp16" or args.pipeline_device or args.weight_mode != "resident"):\n            raise ValueError("Residual MTP draft requires an explicit artifact and resident FP16 non-PP target")\n        if args.mtp_recovery != "replay" and (',1)
s=s.replace('QwenMTPWeights.from_ninfer(NInferArtifact(args.mtp_artifact)),\n                )','QwenMTPWeights.from_ninfer(NInferArtifact(args.mtp_artifact)),\n                    draft_projection_profile=args.mtp_draft_projection_profile,\n                )',1)
s=s.replace('mtp_draft_projection_profile=args.dtype\n                    if args.mtp_artifact\n                    else None,','mtp_draft_projection_profile=info.get("mtp_draft_projection_profile", args.mtp_draft_projection_profile if args.dtype == "fp16" else args.dtype) if args.mtp_artifact else None,')
p.write_text(s)
p=root/'tests/test_qwen_mtp.py';s=p.read_text().replace('*, profile="fp16"):', '*, profile="fp16", draft_profile="fp16"):');s=s.replace('NativeQwenMTPModel.from_q8(target, mtp_weights, reserve_bytes=0)','NativeQwenMTPModel.from_q8(target, mtp_weights, reserve_bytes=0, draft_projection_profile=draft_profile)');p.write_text(s)
p=root/'tests/test_qwen_mtp_batch.py';s=p.read_text().replace('profile="q8-residual", tied=False):','profile="q8-residual", tied=False, draft_profile="fp16"):');s=s.replace('NativeQwenMTPModel.from_q8(target, b, reserve_bytes=0)','NativeQwenMTPModel.from_q8(target, b, reserve_bytes=0, draft_projection_profile=draft_profile)');p.write_text(s)
