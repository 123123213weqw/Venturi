"""CPU-only complete text Q4 bundle export; no GPU or Rust compilation."""
import argparse,json,time,resource
from pathlib import Path
import torch
from benchmark_flash_next_full import checked_payloads
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_q4_bundle import export_q4_text

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--reuse-archive');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter();cpu=time.process_time()
    proof,manifest=checked_payloads(Path(a.checkpoint));cp=inspect_checkpoint(a.checkpoint)
    result=export_q4_text(cp,a.output,source_identity=dict(revision=manifest['hf_revision'],manifest_sha256=proof['manifest_sha256'],decode_dtype='FP16'),reuse_archive=a.reuse_archive,workers=a.workers)
    print(json.dumps(dict(result=result,seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)))

if __name__=='__main__':main()
