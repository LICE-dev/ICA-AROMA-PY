import numpy as np
import pytest

from ica_aroma_py.services.ICA_AROMA_functions import classification


@pytest.mark.parametrize(
    ("hfc", "expected_indices", "expected_file_contents"),
    [
        (np.array([0.10, 0.10, 0.10]), [], ""),
        (np.array([0.10, 0.50, 0.10]), [1], "2"),
        (np.array([0.50, 0.10, 0.60]), [0, 2], "1,3"),
    ],
    ids=("no-motion-components", "one-motion-component", "multiple-motion-components"),
)
def test_classification_returns_a_one_dimensional_index_array(
    tmp_path, hfc, expected_indices, expected_file_contents
):
    """Classification keeps zero-based arrays and one-based output files stable."""
    component_count = len(hfc)

    motion_ics = classification(
        str(tmp_path),
        np.zeros(component_count),
        np.zeros(component_count),
        hfc,
        np.zeros(component_count),
    )

    assert motion_ics.ndim == 1
    assert motion_ics.tolist() == expected_indices
    assert (tmp_path / "classified_motion_ICs.txt").read_text() == expected_file_contents


@pytest.mark.parametrize(
    ("hfc", "expected_motion_ics"),
    [
        (np.array([0.10, 0.10, 0.10]), []),
        (np.array([0.10, 0.50, 0.10]), [2]),
        (np.array([0.50, 0.10, 0.60]), [1, 3]),
    ],
    ids=("no-motion-components", "one-motion-component", "multiple-motion-components"),
)
def test_aroma_classification_exposes_motion_components_as_a_nipype_list(
    tmp_path, hfc, expected_motion_ics
):
    """The Nipype interface accepts every classification cardinality without FSL."""
    pytest.importorskip("nipype")
    from ica_aroma_py.services.ICA_AROMA_nodes import AromaClassification

    component_count = len(hfc)
    result = AromaClassification(
        max_rp_corr=np.zeros(component_count),
        edge_fract=np.zeros(component_count),
        HFC=hfc,
        csf_fract=np.zeros(component_count),
    ).run(cwd=str(tmp_path))

    assert result.outputs.motion_ics == expected_motion_ics
