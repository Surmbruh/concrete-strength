from dataclasses import dataclass


@dataclass(frozen=True)
class Version():
    major: int = 0
    minor: int = 1
    patch: int = 0


VERSION: Version = Version()
