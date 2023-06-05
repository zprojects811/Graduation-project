from stable_baselines3.common.callbacks import CheckpointCallback


class TensorboardCallback(CheckpointCallback):
    """
    Custom callback for plotting Travel Time in tensorboard.
    """

    def __init__(self,
                 save_path="../models/",
                 name_prefix="rl_model"):
        super().__init__(50000, verbose=2, save_path=save_path, name_prefix=name_prefix,
                         save_vecnormalize=True)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        infos = self.locals.get('infos')
        sum = 0
        for info in infos:
            sum += info['avg_travel_time']
        avg = sum / len(infos)
        self.logger.record("avg_travel_time", avg)
        return True
