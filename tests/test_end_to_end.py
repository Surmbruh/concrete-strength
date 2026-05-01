"""Smoke tests: package imports cleanly and CLI parser builds."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from materialgen.cli import _build_parser, main as cli_main


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


class CliSmokeTest(unittest.TestCase):
    def test_all_subcommands_register(self) -> None:
        """The CLI exposes the five pipeline stages."""

        parser = _build_parser()
        # argparse stores subcommands in the choices of the subparsers action.
        sub_action = next(a for a in parser._subparsers._actions if a.dest == "command")
        self.assertEqual(
            set(sub_action.choices),
            {
                "train_surrogate",
                "train_neat",
                "make_neat_to_bnn",
                "batch-predict",
                "complex-check",
            },
        )

    def test_each_stage_has_example_config(self) -> None:
        """Every stage has a checked-in example JSON config."""

        for name in (
            "forward.json",
            "backward.json",
            "make_neat_to_bnn.json",
            "batch.json",
            "complex_check.json",
        ):
            path = EXAMPLES / name
            self.assertTrue(path.exists(), f"missing example: {path}")
            json.loads(path.read_text(encoding="utf-8"))  # parses as JSON

    def test_help_runs(self) -> None:
        """`main.py --help` exits cleanly without traceback."""

        with self.assertRaises(SystemExit) as exc:
            cli_main(["--help"])
        self.assertEqual(exc.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
