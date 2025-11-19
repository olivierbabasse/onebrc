use core::f32;
use memchr::memrchr;
use rustc_hash::FxHashMap;
use std::fmt::Display;
use std::hash::Hash;
use std::hint::assert_unchecked;
use std::slice::from_raw_parts;
use std::str::FromStr;
use std::{
    env::args,
    fs::File,
    io::{self, BufWriter, Write},
    process::Command,
    time::Instant,
};

#[cfg(target_feature = "avx2")]
use std::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi8,
};

#[derive(Eq, PartialOrd)]
struct Name {
    ptr: *const u8,
    len: u8,
}

impl Name {
    #[cfg(target_feature = "avx2")]
    #[target_feature(enable = "avx2")]
    fn eq_internal(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        let s = unsafe { _mm256_loadu_si256(self.ptr as *const __m256i) };
        let o = unsafe { _mm256_loadu_si256(other.ptr as *const __m256i) };
        let mask = (1 << self.len) - 1;
        let diff = _mm256_movemask_epi8(_mm256_cmpeq_epi8(s, o)) as u32;
        diff & mask == mask
    }

    #[cfg(not(target_feature = "avx2"))]
    fn eq_internal(&self, other: &Self) -> bool {
        let self_slice = unsafe { from_raw_parts(self.ptr, self.len as usize) };
        let other_slice = unsafe { from_raw_parts(other.ptr, other.len as usize) };
        self_slice == other_slice
    }
}

impl PartialEq for Name {
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.eq_internal(other) }
    }
}

impl Ord for Name {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_slice = unsafe { from_raw_parts(self.ptr, self.len as usize) };
        let other_slice = unsafe { from_raw_parts(other.ptr, other.len as usize) };
        self_slice.cmp(&other_slice)
    }
}

impl Hash for Name {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.len >= 4 {
            let ptr = self.ptr as *const u32;
            unsafe { ptr.read_unaligned() }.hash(state);
        } else {
            let mut num = 0u64;
            for c in unsafe { from_raw_parts(self.ptr, self.len as usize) } {
                num = (num << 8) + *c as u64;
            }
            num.hash(state)
        }
    }
}

impl Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slice = unsafe { from_raw_parts(self.ptr, self.len as usize) };
        let s = String::from_str(std::str::from_utf8(slice).unwrap()).unwrap();
        write!(f, "{s}")
    }
}

unsafe impl Send for Name {}

struct Entry {
    count: u32,
    min: i32,
    max: i32,
    sum: i32,
}

impl Entry {
    fn new(value: i32) -> Self {
        Self {
            count: 1,
            min: value,
            max: value,
            sum: value,
        }
    }

    fn add_value(&mut self, value: i32) {
        self.count += 1;
        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    fn min(&self) -> f32 {
        self.min as f32 / 10.0
    }
    fn max(&self) -> f32 {
        self.max as f32 / 10.0
    }
    fn average(&self) -> f32 {
        (self.sum as f32 / self.count as f32) / 10.0
    }
}

fn parse_value(mut text: &[u8]) -> i32 {
    unsafe {
        assert_unchecked(text.len() >= 3);
    }
    let negative = text[0] == b'-';
    if negative {
        text = &text[1..];
    }

    unsafe {
        assert_unchecked(text.len() >= 3);
    }
    let tens = [b'0', text[0]][(text.len() > 3) as usize] as i32;
    let ones = (text[text.len() - 3]) as i32;
    let tenths = (text[text.len() - 1]) as i32;
    let abs_val = tens * 100 + ones * 10 + tenths - 111 * b'0' as i32;
    if negative { -abs_val } else { abs_val }
}

#[cfg(not(target_feature = "avx2"))]
fn read_line(mut data: &[u8]) -> (&[u8], Name, &[u8]) {
    use memchr::memchr;
    let name: &[u8];
    let value: &[u8];
    (name, data) = data.split_at(memchr(b';', &data[3..]).unwrap() + 3);
    data = &data[1..];
    (value, data) = data.split_at(memchr(b'\n', &data[3..]).unwrap() + 3);
    data = &data[1..];
    (
        data,
        Name {
            ptr: name.as_ptr(),
            len: name.len() as u8,
        },
        value,
    )
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
fn read_line(data: &[u8]) -> (&[u8], Name, &[u8]) {
    let separator = _mm256_set1_epi8(b';' as i8);
    let eol = _mm256_set1_epi8(b'\n' as i8);
    let line = unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) };
    let separator_pos =
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(line, separator)).trailing_zeros() as usize;
    let eol_pos = _mm256_movemask_epi8(_mm256_cmpeq_epi8(line, eol)).trailing_zeros() as usize;
    (
        &data[eol_pos + 1..],
        Name {
            ptr: data.as_ptr(),
            len: separator_pos as u8,
        },
        &data[separator_pos + 1..eol_pos],
    )
}

fn process_data_chunk(mut data: &[u8]) -> FxHashMap<Name, Entry> {
    //eprintln!("processing {} bytes", data.len());

    let mut entries = FxHashMap::default();

    while data.len() > 32 {
        let name: Name;
        let value: &[u8];
        (data, name, value) = unsafe { read_line(data) };

        let value = parse_value(value);
        entries
            .entry(name)
            .and_modify(|e: &mut Entry| e.add_value(value))
            .or_insert(Entry::new(value));
    }

    entries
}

fn merge_entries(entries1: &mut FxHashMap<Name, Entry>, entries2: FxHashMap<Name, Entry>) {
    entries2.into_iter().for_each(|(name2, entry2)| {
        entries1
            .entry(name2)
            .and_modify(|entry| {
                entry.min = entry.min.min(entry2.min);
                entry.max = entry.max.max(entry2.max);
                entry.sum += entry2.sum;
                entry.count += entry2.count;
            })
            .or_insert(entry2);
    });
}

fn main() -> io::Result<()> {
    let base = args()
        .nth(1)
        .unwrap_or("data/measurements-1000000000".to_string());
    let filename = format!("{base}.txt");

    let start_time = Instant::now();

    #[allow(unused_mut)]
    let mut file = File::open(filename)?;
    let file_length = file.metadata()?.len() as usize;
    eprintln!("input file length : {file_length} bytes");

    #[cfg(not(miri))]
    let data = {
        use memmap2::{Advice, MmapOptions};

        let data = unsafe { MmapOptions::new().len(file_length + 32).map(&file)? };
        data.advise(Advice::Sequential)?;
        data
    };
    #[cfg(miri)]
    let data = {
        use std::io::Read;

        let mut data: Vec<u8> = Vec::with_capacity(file_length + 32);
        file.read_to_end(&mut data)?;
        data.append(&mut vec![0u8; 32]);
        data
    };

    let mut data = &*data;

    let thread_count = std::thread::available_parallelism()?.get();
    let chunk_size = data.len() / thread_count;
    eprintln!("using {thread_count} threads, chunk size = {chunk_size}");

    let entries = std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for _ in 0..thread_count - 1 {
            let chunk_end = memrchr(b'\n', &data[..chunk_size]).unwrap();
            let chunk = &data[..chunk_end + 33];
            data = &data[chunk_end + 1..];
            threads.push(scope.spawn(|| process_data_chunk(chunk)));
        }
        let mut entries = process_data_chunk(data);
        for t in threads {
            merge_entries(&mut entries, t.join().unwrap());
        }
        entries
    });

    let mut entries = entries.into_iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|e1, e2| e1.0.cmp(&e2.0));

    eprintln!("elapsed time : {}s", start_time.elapsed().as_secs_f64());

    let file = File::create("./results.txt")?;
    let mut writer = BufWriter::new(file);
    write!(writer, "{{")?;
    write!(
        writer,
        "{}",
        entries
            .iter()
            .map(|(name, e)| {
                format!(
                    "{}={:.1}/{:.1}/{:.1}",
                    name.to_string(),
                    e.min(),
                    e.average(),
                    e.max()
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(writer, "}}")?;
    writer.flush()?;
    drop(writer);

    #[cfg(not(miri))]
    {
        let output = format!("{base}.out");
        match Command::new("cmp")
            .arg(output)
            .arg("./results.txt")
            .status()?
            .code()
        {
            Some(0) => eprintln!("results match expected output"),
            _ => eprintln!("*** ERROR : results do not match expected output"),
        }
    }

    Ok(())
}
