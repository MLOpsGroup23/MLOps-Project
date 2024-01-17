from MLOps_Project.train_model import train
from hydra import initialize, compose
from os.path import exists


class TestTrain:
    def test_train(self):
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="config", overrides=["training.max_epochs=1", "architecture.name=Resnet"])

            # Update Configs
            train(cfg, wandb_offline=True)

            assert exists(
                "./models/ResNetModel.ckpt"
            ), "After 1 Epoch, expected a model to be made, but no model was found"
