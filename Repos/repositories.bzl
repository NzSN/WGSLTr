load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def argparse_repo():
    new_git_repository(
        name = "argparse",
        remote = "https://github.com/p-ranav/argparse.git",
        branch = "master",
        build_file = "@wgsltr//:Repos/Builds/BUILD.argparse")


def _repositories_ext_impl(_ctx):
    argparse_repo()


repositories_ext = module_extension(
    implementation = _repositories_ext_impl)
