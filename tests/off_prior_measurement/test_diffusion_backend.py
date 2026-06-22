from scripts.off_prior_measurement.diffusion_backend import training_target


class FakeScheduler:
    def get_velocity(self, latents, noise, timesteps):
        return f"velocity({latents},{noise},{timesteps})"


def test_training_target_uses_noise_for_epsilon_prediction():
    target = training_target(
        scheduler=FakeScheduler(),
        prediction_type="epsilon",
        latents="latents",
        noise="noise",
        timesteps="timestep",
    )

    assert target == "noise"


def test_training_target_uses_scheduler_velocity_for_v_prediction():
    target = training_target(
        scheduler=FakeScheduler(),
        prediction_type="v_prediction",
        latents="latents",
        noise="noise",
        timesteps="timestep",
    )

    assert target == "velocity(latents,noise,timestep)"
