import torch,json
r=dict(cuda_visible_devices="1,0",devices=[torch.cuda.get_device_name(i) for i in range(2)],directions=[])
for src,dst in [(0,1),(1,0)]:
 x=(torch.arange(1024,device=f"cuda:{src}",dtype=torch.float32)*3+17)
 y=torch.full((1024,),-999.,device=f"cuda:{dst}")
 y.copy_(x);torch.cuda.synchronize(src);torch.cuda.synchronize(dst)
 actual=y.cpu();expected=x.cpu()
 row=dict(source_logical_device=src,destination_logical_device=dst,advertised_peer_access=torch.cuda.can_device_access_peer(src,dst),direct_copy_bytes_equal=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)),mismatched_values=int((actual!=expected).sum()),first_expected=expected[:8].tolist(),first_actual=actual[:8].tolist())
 y.copy_(expected,non_blocking=False);torch.cuda.synchronize(dst)
 row["explicit_host_copy_bytes_equal"]=torch.equal(y.cpu().view(torch.uint8),expected.view(torch.uint8));r["directions"].append(row)
print(json.dumps(r,indent=2))
