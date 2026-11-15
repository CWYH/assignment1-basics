"""Build all three ports; fetch and verify the pinned PCRE2 source release."""

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
PCRE2_SHA256 = "8d28d7f2c3b970c3a4bf3776bcbb5adfc923183ce74bc8df1ebaad8c1985bd07"


def run(*command):
    print(subprocess.list2cmdline(list(map(str, command))), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java-home", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    vendor = HERE / "vendor"
    vendor.mkdir(exist_ok=True)
    archive = vendor / "pcre2-10.46.tar.gz"
    if not archive.exists():
        with urllib.request.urlopen(
            "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.46/pcre2-10.46.tar.gz", timeout=60
        ) as source:
            with archive.open("wb") as target:
                shutil.copyfileobj(source, target)
    with archive.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != PCRE2_SHA256:
            raise RuntimeError("PCRE2 archive checksum mismatch")
    if not (vendor / "pcre2-10.46" / "CMakeLists.txt").exists():
        with tarfile.open(archive) as source:
            source.extractall(vendor, filter="data")
    run(
        "cmake",
        "-S",
        HERE / "cpp",
        "-B",
        HERE / "build/cpp",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=gcc",
        "-DCMAKE_CXX_COMPILER=g++",
    )
    run("cmake", "--build", HERE / "build/cpp", "--parallel", args.jobs)
    run("dotnet", "build", HERE / "csharp/BpeBenchmark.csproj", "-c", "Release", "--nologo")
    javac = str(args.java_home / "bin/javac.exe") if args.java_home else shutil.which("javac")
    if javac is None and (java_home := os.environ.get("JAVA_HOME")):
        javac = str(Path(java_home) / "bin/javac.exe")
    if javac is None:
        candidates = sorted((Path.home() / ".jdks").glob("*/bin/javac.exe"))
        if candidates:
            javac = str(candidates[0])
    if javac is None:
        raise RuntimeError("Set JAVA_HOME, pass --java-home, or add javac to PATH")
    (HERE / "build/java").mkdir(parents=True, exist_ok=True)
    run(javac, "-encoding", "UTF-8", "-d", HERE / "build/java", HERE / "java/BpeBenchmark.java")


if __name__ == "__main__":
    main()
