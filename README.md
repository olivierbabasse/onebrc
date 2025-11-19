Using datafiles from https://huggingface.co/datasets/nietras/1brc.data

Ubuntu 22.04 / i7-9700K CPU @ 3.60GHz / 32 GB RAM

cat data/measurements-1000000000.txt > /dev/null : 1.233s

naive implementation, default release build options : 96.503s
naive implementation, codegen-units, lto, panic, target native, debug info : 78.692s
using Vec of u8 instead of Strings and parsing as i32 : 60.295s
names as 32-u8 array and custom hashing : 38.124s
using FxHashMap instead of HashMap : 31.770s
mmap input file instead of read_to_end() into Vec : 25.954s
using memchr() instead of position() : 22.198s
using references instead of copying names from buffer : 20.279s
madvise, small line split optimizations : 17.773s
line splitting using avx2 : 15.346s