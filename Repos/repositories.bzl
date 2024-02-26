load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def argparse_repo():
    new_git_repository(
        name = "argparse",
        remote = "https://github.com/p-ranav/argparse.git",
        branch = "master",
        build_file = "@wgsltr//:Repos/Builds/BUILD.argparse")

def antlr4_repo():
    http_archive(
        name = "antlr4_runtime",
        urls =["https://www.antlr.org/download/antlr4-cpp-runtime-4.13.1-source.zip"],
        build_file = "@wgsltr//:Repos/Builds/BUILD.antlr4")


def _repositories_ext_impl(_ctx):
    argparse_repo()
    antlr4_repo()


repositories_ext = module_extension(
    implementation = _repositories_ext_impl)
