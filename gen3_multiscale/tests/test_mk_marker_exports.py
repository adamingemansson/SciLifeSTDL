import json

from PIL import Image

from gen3_multiscale.scripts.export_hest_he_overviews import _white_background
from gen3_multiscale.scripts.plot_mk_tissue_marker_panel import _resolve_panel


def test_he_export_replaces_scanner_black_but_keeps_dark_tissue():
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (0, 0, 0))
    image.putpixel((1, 0), (20, 2, 25))
    result = _white_background(image)
    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((1, 0)) == (20, 2, 25)


def test_marker_panel_filters_missing_genes_and_matches_organs_case_insensitively(tmp_path):
    path = tmp_path / "markers.json"
    path.write_text(json.dumps({
        "common": ["EPCAM", "MISSING"],
        "by_organ": {"Kidney": ["UMOD", "ABSENT"]},
        "hard_immune_controls": ["IGKC"],
    }))
    result = _resolve_panel(path, ["EPCAM", "UMOD", "IGKC"], {"kidney"})
    assert result["common"] == ["EPCAM", "IGKC"]
    assert result["by_organ"]["kidney"] == ["UMOD"]
    assert result["missing"]["common"] == ["MISSING"]
    assert result["missing"]["kidney"] == ["ABSENT"]
