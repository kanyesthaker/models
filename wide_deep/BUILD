py_library(
    name = "wd_distributed_train",
    srcs = [
        "wd_distributed_train.py",
    ],
    deps = [
        ":dnn_linear_combined",
    ]
)

py_library(
    name = "dnn_linear_combined",
    srcs = [
        "dnn_linear_combined.py"
    ],
)

py_binary(
    name = "wide_deep_distributed",
    srcs = [
        "wide_deep_distributed.py",
    ],
    deps = [
        ":wd_distributed_train",
    ],
)