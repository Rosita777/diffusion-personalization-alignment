from PIL import Image

from scripts.off_prior_measurement.config import SubjectSpec
from scripts.off_prior_measurement.generate_controls import build_control_manifest


def test_build_control_manifest_creates_expected_rows(tmp_path):
    subject = SubjectSpec(
        subject_id="dog",
        hf_subset="dog",
        class_name="dog",
        class_prompt="a photo of a dog",
        class_context_prompt="a photo of a dog in a natural scene",
        hard_control_prompt="a photo of a dog under dramatic stage lighting",
    )
    easy_path = tmp_path / "generated" / "dog" / "easy" / "seed_0000.png"
    hard_path = tmp_path / "generated" / "dog" / "hard" / "seed_0000.png"
    easy_path.parent.mkdir(parents=True)
    hard_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(easy_path)
    Image.new("RGB", (8, 8), "black").save(hard_path)

    manifest = build_control_manifest(
        subjects=[subject],
        generated_root=tmp_path / "generated",
        conditionings=["null", "class", "class_context"],
    )

    assert len(manifest) == 6
    assert set(manifest["source_group"]) == {"base_easy_control", "base_hard_control"}
    assert set(manifest["reference_regime"]) == {"easy_control", "hard_control"}
    assert set(manifest["hardness_axis"]) == {"none", "clutter_background"}
    assert set(manifest["variant_id"]) == {""}
    assert set(manifest["conditioning_key"]) == {"null", "class", "class_context"}
