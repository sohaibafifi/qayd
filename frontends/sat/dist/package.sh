#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$script_dir/../../.." && pwd)

if [ -n "${TARGET:-}" ]; then
  target=$TARGET
  target_args="--target $TARGET"
  release_dir="$root/target/$TARGET/release"
else
  target=$(rustc -vV | awk '/host:/ { print $2 }')
  target_args=""
  release_dir="$root/target/release"
fi
version=$(cd "$root" && cargo pkgid -p qayd-sat | sed 's/.*#//' | sed 's/.*@//')
out_root="$root/target/sat-competition"
bundle="qayd-sat-${version}-${target}"
stage="$out_root/$bundle"

mkdir -p "$out_root"
rm -rf "$stage"
mkdir -p "$stage"

(cd "$root" && cargo build --release -p qayd-sat $target_args)

bin="$release_dir/qayd-sat"
if [ ! -x "$bin" ]; then
  echo "qayd-sat release binary not found at $bin" >&2
  exit 1
fi

cp "$bin" "$stage/qayd-sat"
cp "$root/LICENSE" "$stage/LICENSE"
cp "$script_dir/README.md" "$stage/README.md"
cp "$script_dir/smoke.sh" "$stage/smoke.sh"
chmod +x "$stage/qayd-sat" "$stage/smoke.sh"

cat > "$stage/qayd-sat-competition" <<'EOF'
#!/usr/bin/env sh
set -eu
dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$dir/qayd-sat" --competition --native --preprocess basic "$@"
EOF
chmod +x "$stage/qayd-sat-competition"

archive="$out_root/$bundle.tar.gz"
checksum="$archive.sha256"
rm -f "$archive" "$checksum"
(cd "$out_root" && tar -czf "$archive" "$bundle")
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$archive" > "$checksum"
elif command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$archive" > "$checksum"
else
  echo "warning: no sha256 tool found" >&2
fi

echo "$archive"
