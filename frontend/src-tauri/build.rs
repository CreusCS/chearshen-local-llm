use std::path::PathBuf;

fn main() {
  tauri_build::build();

  let protoc_path = protoc_bin_vendored::protoc_bin_path()
    .expect("Unable to locate bundled protoc binary");
  std::env::set_var("PROTOC", protoc_path);

  let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
  let proto_dir = manifest_dir.join("../../proto");
  let proto_file = proto_dir.join("video_analyzer.proto");

  println!("cargo:rerun-if-changed={}", proto_file.display());

  tonic_build::configure()
    .build_server(false)
    .compile(
      &[proto_file.to_str().expect("proto path utf-8")],
      &[proto_dir.to_str().expect("includes path utf-8")],
    )
    .expect("Failed to compile proto definitions");
}
