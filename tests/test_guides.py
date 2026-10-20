import json

import pytest

from core_deepinfer.cli import main
from core_deepinfer.guide import guide_catalog, read_guide


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("chapter", ["00", "01", "02", "03", "04", "05", "06"])
def test_each_bundled_guide_has_a_title_code_and_its_language_companion(chapter, language):
    text = read_guide(chapter, language=language)
    assert text.startswith("# ")
    assert "```" in text
    assert "[中文]" in text if language == "en" else "[English]" in text
    assert len(text) > 500


@pytest.mark.parametrize("language", ["en", "zh"])
def test_catalog_titles_are_read_from_the_same_packaged_markdown(language):
    catalog = guide_catalog(language=language)
    assert [entry["chapter"] for entry in catalog] == [f"{number:02}" for number in range(7)]
    for entry in catalog:
        assert entry["title"] == read_guide(entry["chapter"], language=language).splitlines()[0][2:]


@pytest.mark.parametrize("chapter", ["07", "../README", "", "01-request-loop"])
def test_unknown_chapters_are_explicit_errors(chapter):
    with pytest.raises(ValueError, match="chapter"):
        read_guide(chapter)


def test_unknown_language_is_an_explicit_error():
    with pytest.raises(ValueError, match="language"):
        read_guide("01", language="fr")


@pytest.mark.parametrize("language", ["en", "zh"])
def test_cli_prints_the_complete_requested_guide(language, capsys):
    assert main(["guide", "03", "--lang", language]) == 0
    assert capsys.readouterr().out == read_guide("03", language=language).rstrip() + "\n"


def test_guide_without_a_chapter_reads_the_index(capsys):
    assert main(["guide"]) == 0
    assert capsys.readouterr().out == read_guide().rstrip() + "\n"


@pytest.mark.parametrize("language", ["en", "zh"])
def test_cli_lists_all_guide_titles_as_json(language, capsys):
    assert main(["guide", "--list", "--lang", language, "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"language": language, "guides": guide_catalog(language=language)}


def test_cli_emits_a_complete_markdown_document_as_json(capsys):
    assert main(["guide", "02", "--lang", "zh", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"chapter": "02", "language": "zh", "markdown": read_guide("02", language="zh")}


@pytest.mark.parametrize("arguments", [["guide", "07"], ["guide", "--lang", "fr"], ["demo", "01"]])
def test_invalid_guide_arguments_have_a_cli_error(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
