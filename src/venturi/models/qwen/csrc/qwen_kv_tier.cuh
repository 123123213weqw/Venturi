// Opt-in exact-byte KV tier. Included inside namespace vr, after Buffer/Stream.
// VMM preserves contiguous attention operands; this is NOT a paged-attention
// kernel. Only the current (request, full-attention layer) must fit in VRAM.
#pragma once
struct KVTier {
    struct Entry;
    struct Page { CUmemGenericAllocationHandle handle=0; };
    int device;cudaStream_t stream;void* staging;std::uint64_t staging_bytes,reserve,gpu_limit,ram_limit,granule=0;
    CUmemAllocationProp prop{};CUmemAccessDesc access{};
    std::mutex mutex;bool poisoned=false;
    std::vector<std::unique_ptr<Page>> pool;
    std::vector<Page*> free_pages;
    std::vector<Entry*> entries;
    std::uint64_t ram_bytes=0,serial=0,evictions=0,reloads=0,d2h=0,h2d=0,ooms=0,preflights=0,peak_gpu=0,peak_ram=0;
    void driver(CUresult r) {
        if(r!=CUDA_SUCCESS){poisoned=true;const char* s=nullptr;cuGetErrorString(r,&s);throw std::runtime_error(std::string("native KV VMM: ")+(s?s:"driver error"));}
    }
    std::uint64_t round(std::uint64_t n) const {return n?mul(1+(n-1)/granule,granule):0;}
    struct Entry {
        KVTier& owner;CUdeviceptr base=0;std::uint64_t plane=0,row=0,valid=0,planned=0,last=0,host_plane=0;
        bool active=false;std::uint64_t mapped=0;
        std::array<std::vector<Page*>,2> pages;
        std::unique_ptr<unsigned char[]> host;
        Entry(KVTier& o,std::uint64_t bytes,std::uint64_t row_bytes):owner(o),plane(o.round(bytes)),row(row_bytes) {
            // Prepare all host metadata before reserving address space/registering.
            for(auto& p:pages)p.resize(plane/o.granule,nullptr);
            std::lock_guard<std::mutex> lock(o.mutex);
            o.driver(cuMemAddressReserve(&base,mul(plane,2),o.granule,0,0));
            try {o.entries.push_back(this);}
            catch(...){cuMemAddressFree(base,mul(plane,2));throw;}
        }
        void* ptr(int kind=0,std::uint64_t offset=0) const {return reinterpret_cast<void*>(base+mul(kind,plane)+offset);}
        ~Entry(){owner.remove(*this);}
    };
    KVTier(int d,cudaStream_t s,void* tmp,std::uint64_t tmpbytes,std::uint64_t r,std::uint64_t gpu,std::uint64_t ram):device(d),stream(s),staging(tmp),staging_bytes(tmpbytes),reserve(r),gpu_limit(gpu),ram_limit(ram) {
        int supported=0;driver(cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,d));
        require(supported,"VRAM-first KV requires CUDA virtual memory management");
        prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=d;
        std::size_t g=0;driver(cuMemGetAllocationGranularity(&g,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));granule=g;
        access.location=prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        require(granule && staging && staging_bytes,"invalid native KV tier staging/granularity");
    }
    std::unique_ptr<Entry> create(std::uint64_t bytes,std::uint64_t row){return std::make_unique<Entry>(*this,bytes,row);}
    std::uint64_t physical()const{return mul(pool.size(),granule);}
    void transfer(void* gpu,void* cpu,std::uint64_t n,bool upload) {
        for(std::uint64_t start=0;start<n;){const auto count=std::min(staging_bytes,n-start);
            if(upload)std::memcpy(staging,static_cast<char*>(cpu)+start,count);
            check(cudaMemcpyAsync(upload?static_cast<char*>(gpu)+start:staging,upload?staging:static_cast<char*>(gpu)+start,count,upload?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            if(!upload)std::memcpy(static_cast<char*>(cpu)+start,staging,count);
            start+=count;
        }
        if(upload)h2d=add(h2d,n);else d2h=add(d2h,n);
    }
    void unmap(Entry& e) {
        for(int k=0;k<2;++k)for(std::uint64_t p=0;p<e.pages[k].size();++p)if(e.pages[k][p]){driver(cuMemUnmap(e.base+mul(k,e.plane)+mul(p,granule),granule));free_pages.push_back(e.pages[k][p]);e.pages[k][p]=nullptr;}
        e.mapped=0;
    }
    void evict(Entry& e) {
        require(e.mapped>0,"native KV victim not resident");
        check(cudaStreamSynchronize(stream));
        if(e.valid){require(e.host && e.host_plane>=e.valid,"KV spill backing was not admitted before forward");
            for(int k=0;k<2;++k)transfer(e.ptr(k),e.host.get()+mul(k,e.host_plane),e.valid,false);}
        unmap(e);++evictions;
    }
    // Admission occurs before embedding/GDN mutation. No implicit forward retry.
    void prepare(const std::vector<std::pair<Entry*,std::uint64_t>>& goals) {
        std::lock_guard<std::mutex> lock(mutex);require(!poisoned,"KV VMM driver failure: close and recreate model; do not retry recurrent forward");++preflights;
        for(auto* e:entries){e->planned=e->valid;e->active=false;}
        for(const auto& g:goals){require(g.first && g.second<=g.first->plane,"KV planned prefix exceeds virtual capacity");g.first->planned=std::max(g.first->planned,g.second);g.first->active=true;}
        std::uint64_t needed=0,largest=0;
        for(const auto* e:entries){const auto n=mul(round(e->planned)/granule,2);needed=add(needed,n);largest=std::max(largest,n);}
        std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
        const auto available=add(physical(),free>reserve?free-reserve:0);
        const auto budget=(gpu_limit?std::min(gpu_limit,available):available)/granule;
        require(largest<=budget,"VRAM-first KV budget cannot fit one request/layer prefix; reduce context or batch/offload scope");
        // Release reset/high-water mappings above the admitted logical prefix.
        // Otherwise an old oversized entry could force an unnecessary spill.
        bool trim=false;for(auto* e:entries)if(e->mapped>round(e->planned)/granule)trim=true;
        if(trim){check(cudaStreamSynchronize(stream));for(auto* e:entries){const auto n=round(e->planned)/granule;
            for(int k=0;k<2;++k)for(auto p=n;p<e->mapped;++p){driver(cuMemUnmap(e->base+mul(k,e->plane)+mul(p,granule),granule));free_pages.push_back(e->pages[k][p]);e->pages[k][p]=nullptr;}e->mapped=std::min(e->mapped,n);}}
        const auto target=std::min(needed,budget);
        // Reserve host metadata before any driver allocation, so push_back cannot
        // leak a handle or fail later while a recurrent forward is in progress.
        pool.reserve(std::max<std::uint64_t>(pool.size(),target));free_pages.reserve(std::max<std::uint64_t>(pool.size(),target));
        while(pool.size()<target){auto p=std::make_unique<Page>();const auto r=cuMemCreate(&p->handle,granule,&prop,0);
            if(r==CUDA_ERROR_OUT_OF_MEMORY){++ooms;break;}driver(r);free_pages.push_back(p.get());pool.push_back(std::move(p));}
        peak_gpu=std::max(peak_gpu,physical());
        require(pool.size()>=largest,"native KV physical allocation OOM before forward; state unchanged");
        if(needed>pool.size()){
            // RAM is allocated ONLY under actual/configured residency pressure.
            // Pre-admit every possible victim, including cold/paused requests.
            std::uint64_t final=ram_bytes;
            for(const auto* e:entries){const auto n=round(e->planned);if(n>e->host_plane)final=add(final,mul(n-e->host_plane,2));}
            require(final<=ram_limit,"VRAM-first KV RAM budget exhausted before forward; state unchanged");
            for(auto* e:entries){const auto n=round(e->planned);if(n<=e->host_plane)continue;
                // Include realloc's temporary old+new footprint in the quota.
                require(add(ram_bytes,mul(n,2))<=ram_limit,"KV RAM growth peak exceeds budget before forward");
                std::unique_ptr<unsigned char[]> h(new unsigned char[mul(n,2)]);
                if(e->host && !e->mapped && e->valid)for(int k=0;k<2;++k)std::memcpy(h.get()+mul(k,n),e->host.get()+mul(k,e->host_plane),e->valid);
                ram_bytes=add(ram_bytes,mul(n-e->host_plane,2));e->host=std::move(h);e->host_plane=n;peak_ram=std::max(peak_ram,ram_bytes);
            }
        }
    }
    void ensure(Entry& e,std::uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);require(!poisoned,"KV VMM driver failure: recreate model");const auto n=round(bytes)/granule;e.last=++serial;
        require(bytes<=e.plane && n<=pool.size()/2,"native KV prefix not admitted");
        if(e.mapped>=n)return;
        check(cudaStreamSynchronize(stream));
        const bool reload=!e.mapped && e.valid;
        const auto extra=mul(n-e.mapped,2);
        while(free_pages.size()<extra){Entry* victim=nullptr;
            for(auto* other:entries)if(other!=&e && other->mapped && (!victim || (other->active!=victim->active?!other->active:other->last<victim->last)))victim=other;
            require(victim,"native KV residency admission invariant failed");evict(*victim);
        }
        for(int k=0;k<2;++k)for(auto p=e.mapped;p<n;++p){auto* page=free_pages.back();const auto addr=e.base+mul(k,e.plane)+mul(p,granule);
            driver(cuMemMap(addr,granule,0,page->handle,0));
            // Track ownership as soon as mapping succeeds, even if access setup
            // fails. A poisoned model cannot reuse this page; teardown owns it.
            free_pages.pop_back();e.pages[k][p]=page;
            driver(cuMemSetAccess(addr,granule,&access,1));
        }
        e.mapped=n;
        if(reload){require(e.host && e.host_plane>=e.valid,"native KV reload backing absent");for(int k=0;k<2;++k)transfer(e.ptr(k),e.host.get()+mul(k,e.host_plane),e.valid,true);++reloads;}
    }
    void written(Entry& e,std::uint64_t bytes){std::lock_guard<std::mutex> lock(mutex);require(bytes<=mul(e.mapped,granule),"KV write outside committed prefix");e.valid=bytes;}
    void snapshot(Entry& e,int k,void* target,std::uint64_t bytes){std::lock_guard<std::mutex> lock(mutex);require(!poisoned,"KV VMM driver failure: recreate model");require(bytes<=e.valid,"KV snapshot exceeds valid prefix");check(cudaStreamSynchronize(stream));
        if(bytes){if(e.mapped){check(cudaMemcpyAsync(target,e.ptr(k),bytes,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));}else{require(e.host && bytes<=e.host_plane,"KV snapshot backing absent");std::memcpy(target,e.host.get()+mul(k,e.host_plane),bytes);}}}
    void reset(Entry& e){std::lock_guard<std::mutex> lock(mutex);require(!poisoned,"KV VMM driver failure: recreate model");check(cudaStreamSynchronize(stream));e.valid=e.planned=0;e.host.reset();ram_bytes-=mul(e.host_plane,2);e.host_plane=0;e.active=false;}
    void remove(Entry& e) noexcept {try{std::lock_guard<std::mutex> lock(mutex);
        // Unregister before fallible CUDA cleanup: never leave a dangling Entry*
        // for another cohort after a context/driver teardown error.
        entries.erase(std::remove(entries.begin(),entries.end(),&e),entries.end());ram_bytes-=mul(e.host_plane,2);
        Device d(device);check(cudaStreamSynchronize(stream));unmap(e);driver(cuMemAddressFree(e.base,mul(e.plane,2)));
    }catch(...){/* CUDA teardown failure must not throw from a destructor. */}}
    std::array<std::uint64_t,18> info(){std::lock_guard<std::mutex> lock(mutex);std::uint64_t mapped=0,virt=0,valid=0;for(const auto* e:entries){mapped=add(mapped,mul(mul(e->mapped,granule),2));virt=add(virt,mul(e->plane,2));valid=add(valid,mul(e->valid,2));}
        return {1,granule,physical(),mapped,virt,ram_bytes,valid,evictions,reloads,d2h,h2d,ooms,preflights,peak_gpu,peak_ram,gpu_limit,ram_limit,entries.size()};}
    ~KVTier(){Device d(device);for(const auto& p:pool)if(p->handle)(void)cuMemRelease(p->handle);}
};
