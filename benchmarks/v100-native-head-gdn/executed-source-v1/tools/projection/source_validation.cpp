// Remote-only CPU microbenchmark of the existing metadata verification pair.
// No mmap payload reads, mutation, source guard bypass, or model execution.
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <stdexcept>

static bool same(const struct stat& a, const struct stat& b) {
    return a.st_dev==b.st_dev && a.st_ino==b.st_ino && a.st_size==b.st_size &&
        a.st_mtim.tv_sec==b.st_mtim.tv_sec && a.st_mtim.tv_nsec==b.st_mtim.tv_nsec &&
        a.st_ctim.tv_sec==b.st_ctim.tv_sec && a.st_ctim.tv_nsec==b.st_ctim.tv_nsec;
}

int main(int argc, char** argv) {
    if(argc!=2) return 2;
    int fd=::open(argv[1], O_RDONLY|O_CLOEXEC);
    struct stat initial{};
    if(fd<0 || ::fstat(fd,&initial)) return 3;
    constexpr int pairs=100000;
    std::puts("{\"verification_pairs_per_repeat\":100000,\"seconds\":[");
    for(int repeat=0;repeat<4;++repeat) {
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<pairs;++i) {
            struct stat a{},b{};
            if(::fstat(fd,&a) || ::stat(argv[1],&b) || !same(initial,a) || !same(initial,b))
                throw std::runtime_error("Checkpoint identity changed in diagnostic");
        }
        auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::printf("%.12f%s\n",elapsed,repeat==3?"":",");
    }
    ::close(fd);
    std::puts("],\"scope\":\"Untraced warm CPU metadata pairs, not inference critical-path cost\"}");
}
