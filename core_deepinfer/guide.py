"""Read the same bilingual walkthroughs from a checkout or an installed wheel."""

from importlib.resources import files

_CHAPTERS = (
    "00-index",
    "01-request-loop",
    "02-state-recovery",
    "03-speculative-transactions",
    "04-budget-scheduling",
    "05-model-and-replay",
    "06-fork-and-extend",
)


def read_guide(chapter="00", *, language="en"):
    if language not in ("en", "zh"):
        raise ValueError("language must be en or zh")
    names = {name[:2]: name for name in _CHAPTERS}
    if chapter not in names:
        raise ValueError("chapter must be 00 through 06")
    suffix = "_EN.md" if language == "en" else ".md"
    return files("core_deepinfer").joinpath("guides", names[chapter] + suffix).read_text(encoding="utf-8")


def guide_catalog(*, language="en"):
    return [
        {"chapter": name[:2], "title": read_guide(name[:2], language=language).splitlines()[0][2:]}
        for name in _CHAPTERS
    ]
