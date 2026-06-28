from scripts.off_prior_measurement.config import load_config


def test_load_config_parses_smoke_test_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    subject_path = tmp_path / "subjects.yaml"
    subject_path.write_text(
        """
subjects:
  - subject_id: dog
    hf_subset: dog
    class_name: dog
    class_prompt: a photo of a dog
    class_context_prompt: a photo of a dog in a natural scene
    hard_control_prompt: a photo of a dog under dramatic stage lighting in a cluttered room
""".strip(),
        encoding="utf-8",
    )
    config_path.write_text(
        f"""
experiment_name: smoke_test
model_id: runwayml/stable-diffusion-v1-5
prediction_type: epsilon
device: cpu
dtype: float32
resolution: 512
dataset_repo: google/dreambooth
subject_manifest: {subject_path}
cache_dir: data/cache/off_prior_measurement_v0
output_dir: experiments/off_prior_measurement_v0/smoke_test
debug_output_dir: outputs/off_prior_measurement_v0/smoke_test
timesteps: [50, 200, 500, 800, 999]
noise_seeds: [0, 1, 2]
conditionings: ["null", "class", "class_context"]
control_images_per_subject: 2
batch_size: 1
save_debug_tensors: false
dataset_source: github
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.experiment_name == "smoke_test"
    assert config.model_id == "runwayml/stable-diffusion-v1-5"
    assert config.timesteps == [50, 200, 500, 800, 999]
    assert config.noise_seeds == [0, 1, 2]
    assert config.conditionings == ["null", "class", "class_context"]
    assert config.dataset_source == "github"
    assert config.subjects[0].subject_id == "dog"
    assert config.subjects[0].class_prompt == "a photo of a dog"


def test_load_config_parses_ladder_v2_fields(tmp_path):
    subject_path = tmp_path / "subjects.yaml"
    subject_path.write_text(
        """
subjects:
  - subject_id: colorful_sneaker
    hf_subset: colorful_sneaker
    class_name: sneaker
    class_prompt: a photo of a sneaker
    class_context_prompt: a photo of a sneaker on the floor
    hard_control_prompt: a photo of a colorful sneaker under neon light on reflective metal
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment_name: ladder_v2
model_id: data/cache/modelscope/AI-ModelScope/stable-diffusion-v1-5
prediction_type: epsilon
device: cuda
dtype: float16
resolution: 512
dataset_repo: google/dreambooth
dataset_source: github
subject_manifest: {subject_path}
cache_dir: data/cache/off_prior_measurement_v0
output_dir: experiments/off_prior_measurement_v0/ladder_v2
debug_output_dir: outputs/off_prior_measurement_v0/ladder_v2
timesteps: [50, 200]
noise_seeds: [0, 1]
conditionings: ["null", "class", "class_context"]
control_images_per_subject: 2
batch_size: 1
save_debug_tensors: false
hard_reference_variants:
  - crop_large_subject
  - low_light_color_shift
hard_reference_limit_per_subject: 3
reference_images_per_subject: 3
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.experiment_name == "ladder_v2"
    assert config.hard_reference_variants == ["crop_large_subject", "low_light_color_shift"]
    assert config.hard_reference_limit_per_subject == 3
    assert config.reference_images_per_subject == 3


def test_load_config_parses_source_decomp_fields(tmp_path):
    subject_path = tmp_path / "subjects.yaml"
    ordinary_path = tmp_path / "ordinary.yaml"
    subject_path.write_text(
        """
subjects:
  - subject_id: dog
    hf_subset: dog
    class_name: dog
    class_prompt: a photo of a dog
    class_context_prompt: a photo of a dog in a natural scene
    hard_control_prompt: a photo of a dog in a cluttered room
""".strip(),
        encoding="utf-8",
    )
    ordinary_path.write_text(
        """
ordinary_real_controls:
  - class_name: dog
    image_id: dog_real_00
    image_path: data/local_real_controls/dog/dog_real_00.jpg
    source_dataset: local_real_controls
    source_license_note: local research-only placeholder
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment_name: source_decomp_v1
model_id: data/cache/modelscope/AI-ModelScope/stable-diffusion-v1-5
prediction_type: epsilon
device: cuda
dtype: float16
resolution: 512
dataset_repo: google/dreambooth
dataset_source: github_api
subject_manifest: {subject_path}
ordinary_real_manifest: {ordinary_path}
cache_dir: data/cache/off_prior_measurement_v0
output_dir: experiments/off_prior_measurement_v0/source_decomp_v1
debug_output_dir: outputs/off_prior_measurement_v0/source_decomp_v1
timesteps: [50, 200]
noise_seeds: [0, 1]
conditionings: ["class", "class_context"]
control_images_per_subject: 2
batch_size: 1
save_debug_tensors: false
source_decomp_images_per_class: 2
source_decomp_save_debug_tensors: false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.experiment_name == "source_decomp_v1"
    assert config.ordinary_real_manifest == ordinary_path
    assert config.source_decomp_images_per_class == 2
    assert config.source_decomp_save_debug_tensors is False
