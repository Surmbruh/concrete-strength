"""Точка входа в MaterialGen.

    python3 main.py train_neat       --config examples/backward.json
    python3 main.py make_neat_to_bnn --config examples/make_neat_to_bnn.json
"""

from materialgen.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
