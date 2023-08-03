load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")

pytype_strict_binary(
    name = "read_ibr_psv",
    srcs = ["read_ibr_psv.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/etils/epath",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
    ],
)

pytype_strict_binary(
    name = "read_ibr_llf",
    srcs = ["read_ibr_llf.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/etils/epath",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
    ],
)

pytype_strict_binary(
    name = "ibr_psv_to_exr",
    srcs = ["ibr_psv_to_exr.py"],
    deps = [
        "//pyglib/concurrent:parallel",
        "//research/vision/viscam/image_utils/python:exr_to_numpy",
        "//third_party/py/absl:app",
        "//third_party/py/etils/epath",
        "//third_party/py/numpy",
    ],
)
