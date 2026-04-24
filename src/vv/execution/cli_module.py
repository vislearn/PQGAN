from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.utilities.model_summary import ModelSummary
from pytorch_lightning.utilities import rank_zero_only

from vv.logger import MetricsLogger
from vv.logger.image_logger import ValidationImageLogger


class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # Add early stopping
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({"early_stopping.monitor": "val_loss",
                             "early_stopping.patience": 5,
                             "early_stopping.verbose": False,
                             "early_stopping.mode": "min"})
        # Add LR monitor
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults({"lr_monitor.logging_interval": "epoch"})

        # Add model checkpointing behavior
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults({"checkpoint.filename": "{epoch}-{step}-{val_loss:.4f}",
                             "checkpoint.every_n_train_steps": 10000,
                             "checkpoint.save_weights_only": False,
                             "checkpoint.save_top_k": -1})

        # Add image logger
        parser.add_lightning_class_args(ValidationImageLogger, "image_logger")
        parser.set_defaults({"image_logger.num_images": 5,
                             "image_logger.tag_name": "images_in_vs_out"})

        # Add fid kid logger
        parser.add_lightning_class_args(MetricsLogger, "metrics_logger")

        parser.set_defaults({
            "metrics_logger.log_every_n_val_epoch": 20,
            "metrics_logger.fid_config.feature_layer_size": 2048,
            "metrics_logger.kid_config.feature_layer_size": 2048,
            "metrics_logger.is_config.feature_layer_size": 2048,
            "metrics_logger.is_config.splits": 10,
            "metrics_logger.lpips_config.net_type": "alex",
            "metrics_logger.lpips_config.reduction": "mean",
            "metrics_logger.psnr_config.data_range": [0.0, 1.0],
            "metrics_logger.psnr_config.base": 10,
            "metrics_logger.psnr_config.reduction": "elementwise_mean",
            "metrics_logger.psnr_config.dim": None,
        })

        # Add verbose mode argument
        parser.add_argument("--verbose_mode",
                            type=bool,
                            default=False)

        # Link the image size to the layer size of the model
        # TODO: Linking does not work
        """
        parser.link_arguments("data.data_size",
                              "model.init_args.encoder.input_size",
                              apply_on="instantiate")
        parser.link_arguments("data.data_size",
                              "model.init_args.decoder.output_size",
                              apply_on="instantiate")
        """

    def before_fit(self) -> None:
        # Print out the layers of the model
        if self.config.fit.verbose_mode:
            self.__print_model_summary()

    @rank_zero_only
    def __print_model_summary(self) -> None:
        summary = ModelSummary(self.model, max_depth=-1)
        print(summary)
