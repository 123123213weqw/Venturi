// Venturi row-split weight storage validation: one bounded CPU scan of the
// mapped encoded planes at load time. Included inside namespace vr by
// qwen_runtime.cuh, after weight_source.cuh and the geometry helpers. For
// 8-bit codecs the forbidden code -128 is rejected; FP16 scales must be
// finite and sign-clean for every codec.
#pragma once

inline void validate_codes_source(const WeightSource& source,void* staging,std::uint64_t stage_bytes,const VRMatrix& m,int bits,int group) {
    const std::uint64_t code_bytes=codes_bytes(m,bits),scale_bytes=scales_bytes(m,group);
    // One bounded CPU validation at load, not a repeated per-token scan.
    for (std::uint64_t start=0;start<code_bytes;) {
        auto n=std::min(stage_bytes,code_bytes-start);source.read_bytes(staging,m.offset+start,n);
        if (bits==8)
            for (std::uint64_t i=0;i<n;++i) if (static_cast<const unsigned char*>(staging)[i]==128)
                throw std::runtime_error("native Qwen forbidden Q8 code -128");
        start+=n;
    }
    for (std::uint64_t start=0;start<scale_bytes;) {
        auto n=std::min(stage_bytes,scale_bytes-start);source.read_bytes(staging,m.scales+start,n);
        const auto* p=static_cast<const std::uint16_t*>(staging);
        for (std::uint64_t i=0;i<n/2;++i) if ((p[i]&0x7c00)==0x7c00 || ((p[i]&0x8000) && (p[i]&0x7fff)))
            throw std::runtime_error("native Qwen invalid Q8 scale");
        start+=n;
    }
}
