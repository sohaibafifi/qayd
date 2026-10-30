fn main() {
    println!("cargo:rerun-if-changed={}", file!());
    // Python supplies these symbols when loading the extension on macOS.
    // Maturin adds the same flags; plain Cargo builds need them as well.
    if std::env::var_os("CARGO_FEATURE_PYTHON").is_some() && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
