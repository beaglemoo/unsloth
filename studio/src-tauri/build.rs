// aarch64-pc-windows-msvc needs clang on PATH: ring and aws-lc-sys assemble via cc-rs with clang.
fn main() {
    // Baked into the fork build with option_env!; a stale value would repair from a wrong checkout.
    println!("cargo:rerun-if-env-changed=UNSLOTH_FORK_REPO");
    println!("cargo:rerun-if-changed=windows/app-manifest.xml");
    let windows = tauri_build::WindowsAttributes::new()
        .app_manifest(include_str!("windows/app-manifest.xml"));
    tauri_build::try_build(tauri_build::Attributes::new().windows_attributes(windows))
        .expect("failed to build Tauri application resources");
}
