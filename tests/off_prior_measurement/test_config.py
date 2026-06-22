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
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.experiment_name == "smoke_test"
    assert config.model_id == "runwayml/stable-diffusion-v1-5"
    assert config.timesteps == [50, 200, 500, 800, 999]
    assert config.noise_seeds == [0, 1, 2]
    assert config.conditionings == ["null", "class", "class_context"]
    assert config.subjects[0].subject_id == "dog"
    assert config.subjects[0].class_prompt == "a photo of a dog"
