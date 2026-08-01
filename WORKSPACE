workspace(name = "rtp_llm")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_opentelemetry_cpp",
    # v1.21.0 tag commit: b9cf499ff5715433848b316059714b5c59af1f2c
    sha256 = "d020f3aa595a9e0cb8db468c07383e8771744cfe8d0257af4aa721f82c5b4220",
    strip_prefix = "opentelemetry-cpp-b9cf499ff5715433848b316059714b5c59af1f2c",
    patch_args = ["-p1"],
    patches = ["//patches/opentelemetry_cpp:0001-trace-only-otlp-recordable.patch"],
    repo_mapping = {
        "@zlib": "@zlib_archive",
    },
    urls = [
        "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/third_party/opentelemetry-cpp/opentelemetry-cpp-b9cf499ff5715433848b316059714b5c59af1f2c.tar.gz",
        "https://github.com/open-telemetry/opentelemetry-cpp/archive/b9cf499ff5715433848b316059714b5c59af1f2c.tar.gz",
    ],
)

load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("//3rdparty/py:python_configure.bzl", "python_configure")

cuda_configure(name = "local_config_cuda")

rocm_configure(name = "local_config_rocm")

python_configure(name = "local_config_python")

load("//deps:http.bzl", "http_deps")

http_deps()

load("//deps:git.bzl", "git_deps")

git_deps()

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

maybe(
    new_git_repository,
    name = "xgrammar",
    remote = "git@gitlab.alibaba-inc.com:foundation_models/xgrammar_github.git",
    commit = "2ea71da4ccb997a06928c9fb69b99f330da56697",  # v0.2.5
    init_submodules = False,
    patch_cmds = [
        "git submodule update --init --depth=1 3rdparty/dlpack 3rdparty/picojson",
    ],
    build_file = "//3rdparty/xgrammar:xgrammar.BUILD",
)

load("@io_opentelemetry_cpp//bazel:repository.bzl", "opentelemetry_cpp_deps")

opentelemetry_cpp_deps()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("//deps:pip.bzl", "pip_deps")

pip_deps()

load("@pip_cpu_torch//:requirements.bzl", pip_cpu_torch_install_deps = "install_deps")
pip_cpu_torch_install_deps()

load("@pip_arm_torch//:requirements.bzl", pip_arm_torch_install_deps = "install_deps")
pip_arm_torch_install_deps()

load("@pip_ppu_torch//:requirements.bzl", pip_ppu_torch_install_deps = "install_deps")
pip_ppu_torch_install_deps()

load("@pip_gpu_cuda12_torch//:requirements.bzl", pip_gpu_cuda12_torch_install_deps = "install_deps")
pip_gpu_cuda12_torch_install_deps()

load("@pip_gpu_cuda12_9_torch//:requirements.bzl", pip_gpu_cuda12_9_torch_install_deps = "install_deps")
pip_gpu_cuda12_9_torch_install_deps()

load("@pip_gpu_cuda13_torch//:requirements.bzl", pip_gpu_cuda13_torch_install_deps = "install_deps")
pip_gpu_cuda13_torch_install_deps()

load("@pip_cuda12_arm_torch//:requirements.bzl", pip_cuda12_arm_torch_install_deps = "install_deps")
pip_cuda12_arm_torch_install_deps()

load("@pip_cuda13_arm_torch//:requirements.bzl", pip_cuda13_arm_torch_install_deps = "install_deps")
pip_cuda13_arm_torch_install_deps()

load("@pip_gpu_rocm_torch//:requirements.bzl", pip_gpu_rocm_torch_install_deps = "install_deps")
pip_gpu_rocm_torch_install_deps()

load("//:def.bzl", "read_release_version")
read_release_version(name = "release_version")
