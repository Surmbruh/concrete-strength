"""Local helper entry point for quick manual runs of the staged CLI."""

import json

import materialgen.cli as cl


def stage2():
    parser = cl._build_parser()
    argv = ["train_neat",
            "--config", "examples/backward.json",
            "--artifacts-dir", "artifacts"]
    return parser.parse_args(argv)


def stage3():
    parser = cl._build_parser()
    argv = ["make_neat_to_bnn",
            "--config", "examples/make_neat_to_bnn.json",
            "--artifacts-dir", "artifacts"]
    return parser.parse_args(argv)


if __name__ == "__main__":
    handlers = {
        "train_neat": cl._handle_train_neat,
        "make_neat_to_bnn": cl._handle_make_neat_to_bnn,
    }

    args = stage2()
    handlers[args.command](args)
    args = stage3()
    handlers[args.command](args)

    import threading
    import multiprocessing
    print("Живые потоки:", [t.name for t in threading.enumerate()])
    print("Живые процессы:", multiprocessing.active_children())
