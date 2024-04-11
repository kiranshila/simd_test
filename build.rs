fn main() {
    cc::Build::new()
        .file("nanmean.c")
        .flag("-O3")
        .flag("-mavx512f")
        .compile("nanmean");
}
