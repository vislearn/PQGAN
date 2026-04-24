import lightning as L
import torch
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import vv.metrics.configurations as cfg
from vv.metrics.cmmd import CMMDMetric


class MetricsLogger(L.Callback):
    def __init__(self,
                 log_every_n_val_epoch: int = 20,
                 fid_config: cfg.FIDConfig = cfg.FIDConfig(),
                 kid_config: cfg.KIDConfig = cfg.KIDConfig(),
                 is_config: cfg.ISConfig = cfg.ISConfig(),
                 lpips_config: cfg.LPIPSConfig = cfg.LPIPSConfig(),
                 psnr_config: cfg.PSNRConfig = cfg.PSNRConfig(),
                 cmmd_config: cfg.CMMDConfig = cfg.CMMDConfig()) -> None:
        """
        Initialize the Image metrics logger

        Parameters
        ----------
        log_every_n_val_epoch : int
            The frequency of logging metrics during validation epochs.
        fid_config : FIDConfig
            Configuration for the FID metric.
        kid_config : KIDConfig
            Configuration for the KID metric.
        is_config : ISConfig
            Configuration for the IS metric.
        lpips_config : LPIPSConfig
            Configuration for the LPIPS metric.
        psnr_config : PSNRConfig
            Configuration for the PSNR metric.
        cmmd_config : CMMDConfig
            Configuration for the CMMD metric.

        Returns
        -------
        None
        """
        super().__init__()

        self.metrics = MetricCollection({
            name: metric for name, metric, config in [
                ("FID", FrechetInceptionDistance(feature=fid_config.feature_layer_size, normalize=True) if fid_config.active else None, fid_config),
                ("KID", KernelInceptionDistance(feature=kid_config.feature_layer_size, normalize=True) if kid_config.active else None, kid_config),
                ("IS", InceptionScore(feature=is_config.feature_layer_size, splits=is_config.splits, normalize=True) if is_config.active else None, is_config),
                ("LPIPS", LearnedPerceptualImagePatchSimilarity(net_type=lpips_config.net_type, reduction=lpips_config.reduction, normalize=True) if lpips_config.active else None, lpips_config),
                ("PSNR", PeakSignalNoiseRatio(data_range=psnr_config.data_range, base=psnr_config.base, reduction=psnr_config.reduction, dim=psnr_config.dim) if psnr_config.active else None, psnr_config),
                ("CMMD", CMMDMetric() if cmmd_config.active else None, cmmd_config)
            ] if config.active
        })

        self.log_every_n_val_epoch = log_every_n_val_epoch
        self.val_epoch_step_count = 0
        self.logging_active = False

    def on_fit_start(self,
                     trainer: L.Trainer,
                     pl_module: L.LightningModule) -> None:
        """
        Run a dummy validation step to estimate memory usage before training starts.

        Parameters
        ----------
        trainer : lightning.Trainer
            The lightning trainer instance.
        pl_module : lightning.LightningModule
            The lightning module instance.

        Returns
        -------
        None
        """
        # Move metrics to the correct device
        self.metrics.to(pl_module.device)

        if not trainer.datamodule:
            raise ValueError("Trainer does not have a datamodule. Please ensure a datamodule is provided.")

        self.renormalizer = trainer.datamodule.get_renormalizer(target_min=0.0, target_max=1.0)

        batch_size = trainer.datamodule.batch_size
        # Create a dummy batch of data, size doesn't really matter because the model will upscale it
        sample_input = torch.clamp(torch.randn((batch_size, 3, 256, 256), device=pl_module.device), 0, 1)
        sample_output = torch.clamp(torch.randn((batch_size, 3, 256, 256), device=pl_module.device), 0, 1)

        torch.cuda.empty_cache()  # Clear any residual memory
        mem_before = torch.cuda.memory_allocated(torch.cuda.current_device()) / 1e9 if torch.cuda.is_available() else 0

        with torch.no_grad():
            if "FID" in self.metrics:
                self.metrics["FID"].update(sample_input, real=True)
                self.metrics["FID"].update(sample_output, real=False)
            if "KID" in self.metrics:
                self.metrics["KID"].update(sample_input, real=True)
                self.metrics["KID"].update(sample_output, real=False)
            if "IS" in self.metrics:
                self.metrics["IS"].update(sample_output)
            if "LPIPS" in self.metrics:
                self.metrics["LPIPS"].update(sample_input, sample_output)
            if "PSNR" in self.metrics:
                self.metrics["PSNR"].update(sample_input, sample_output)
            if "CMMD" in self.metrics:
                self.metrics["CMMD"].update(sample_input, sample_output)

        mem_after = torch.cuda.memory_allocated(torch.cuda.current_device()) / 1e9 if torch.cuda.is_available() else 0

        if mem_after - mem_before > 1.0:  # If validation adds >1GB usage, warn
            gpu_num = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_num)
            print(f"WARNING: High memory usage on GPU {gpu_num} ({gpu_name})")
            print(f"WARNING: High memory usage detected ({mem_after - mem_before:.2f} GB). Reduce batch size if needed!")

        self.metrics.reset()

    def on_validation_epoch_start(self,
                                  trainer: L.Trainer,
                                  pl_module: L.LightningModule) -> None:
        """
        Called when the validation epoch begins and keeps track of the number of validation steps.
        This is used to determine when to log metrics.

        Parameters
        ----------
        trainer : lightning.Trainer
            The lightning trainer instance.
        pl_module : lightning.LightningModule
            The lightning module instance.

        Returns
        -------
        """
        if not trainer.sanity_checking:
            self.val_epoch_step_count += 1
            self.logging_active = (self.val_epoch_step_count % self.log_every_n_val_epoch == 0)
            if self.logging_active:
                self.metrics.reset()

    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: dict,
                                batch: torch.Tensor,
                                batch_idx: int) -> None:
        """
        Called when the validation batch ends.
        This method updates the metrics with the outputs of the validation step.

        Parameters
        ----------
        trainer : lightning.Trainer
            The lightning trainer instance.
        pl_module : lightning.LightningModule
            The lightning module instance.
        outputs : dict
            The outputs of the validation step.
        batch : torch.Tensor
            The input batch for the validation.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        None
        """
        if self.logging_active:
            # Only data no labels
            real_images = batch[0]
            generated_images = outputs["rec"]
            # Transform to correct range
            normed_real_images = self.renormalizer(real_images)
            normed_generated_images = self.renormalizer(generated_images)

            # This is really bad, but there is no other way to do it
            normed_generated_images = torch.clamp(normed_generated_images, 0, 1)

            # Saftey due to rounding errors
            normed_real_images = torch.clamp(normed_real_images, 0, 1)

            if "FID" in self.metrics:
                self.metrics["FID"].update(normed_real_images, real=True)
                self.metrics["FID"].update(normed_generated_images, real=False)

            if "KID" in self.metrics:
                self.metrics["KID"].update(normed_real_images, real=True)
                self.metrics["KID"].update(normed_generated_images, real=False)

            if "IS" in self.metrics:
                self.metrics["IS"].update(normed_generated_images)

            if "LPIPS" in self.metrics:
                self.metrics["LPIPS"].update(normed_real_images, normed_generated_images)

            if "PSNR" in self.metrics:
                self.metrics["PSNR"].update(normed_real_images, normed_generated_images)

            if "CMMD" in self.metrics:
                self.metrics["CMMD"].update(normed_real_images, normed_generated_images)

    def on_validation_epoch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule) -> None:
        """
        Computes and logs metrics at the end of the validation epoch.

        Parameters
        ----------
        trainer : lightning.Trainer
            The lightning trainer instance.
        pl_module : lightning.LightningModule
            The lightning module instance.

        Returns
        -------
        None
        """
        if self.logging_active:
            if trainer.progress_bar_callback:
                trainer.progress_bar_callback.val_progress_bar.set_description("Computing Metrics...")

            metrics_to_log = {}

            # FID computation
            if "FID" in self.metrics:
                if trainer.progress_bar_callback:
                    trainer.progress_bar_callback.val_progress_bar.set_description("Computing FID...")
                fid_value = self.metrics["FID"].compute()
                metrics_to_log["FID"] = fid_value.item()

            # KID computation
            if "KID" in self.metrics:
                if trainer.progress_bar_callback:
                    trainer.progress_bar_callback.val_progress_bar.set_description("Computing KID...")
                kid_mean, kid_std = self.metrics["KID"].compute()
                metrics_to_log["KID_mean"] = kid_mean.item()
                metrics_to_log["KID_std"] = kid_std.item()

            # Inception Score computation
            if "IS" in self.metrics:
                if trainer.progress_bar_callback:
                    trainer.progress_bar_callback.val_progress_bar.set_description("Computing IS...")
                is_mean, is_std = self.metrics["IS"].compute()
                metrics_to_log["IS_mean"] = is_mean.item()
                metrics_to_log["IS_std"] = is_std.item()

            # LPIPS computation
            if "LPIPS" in self.metrics:
                if trainer.progress_bar_callback:
                    trainer.progress_bar_callback.val_progress_bar.set_description("Computing LPIPS...")
                lpips_value = self.metrics["LPIPS"].compute()
                metrics_to_log["LPIPS"] = lpips_value.item()

            # PSNR computation
            if "PSNR" in self.metrics:
                if trainer.progress_bar_callback:
                    trainer.progress_bar_callback.val_progress_bar.set_description("Computing PSNR...")
                psnr_value = self.metrics["PSNR"].compute()
                metrics_to_log["PSNR"] = psnr_value.item()

            # CMMD computation
            if "CMMD" in self.metrics:
                if trainer.progress_bar_callback:
                    trainer.progress_bar_callback.val_progress_bar.set_description("Computing CMMD...")
                cmmd_value = self.metrics["CMMD"].compute()
                metrics_to_log["CMMD"] = cmmd_value.item()

            # Reset the progress bar description
            if trainer.progress_bar_callback:
                trainer.progress_bar_callback.val_progress_bar.set_description("Validation")

            # Log the values once for rank 0
            if trainer.global_rank == 0:
                step = pl_module.global_step / trainer.world_size
                trainer.logger.log_metrics(metrics_to_log, step=step)

    def on_save_checkpoint(self,
                           trainer: L.Trainer,
                           pl_module: L.LightningModule,
                           checkpoint: dict) -> None:
        """
        Save the state of the callback.

        Parameters
        ----------
        trainer : lightning.Trainer
            The lightning trainer instance.
        pl_module : lightning.LightningModule
            The lightning module instance.
        checkpoint : dict
            The checkpoint dictionary to save the state.

        Returns
        -------
        None
        """
        checkpoint["fidkid_logger"] = {
            "val_epoch_step_count": self.val_epoch_step_count,
        }

    def on_load_checkpoint(self,
                           trainer: L.Trainer,
                           pl_module: L.LightningModule,
                           checkpoint: dict) -> None:
        """
        Restore the state of the callback.

        Parameters
        ----------
        trainer : lightning.Trainer
            The lightning trainer instance.
        pl_module : lightning.LightningModule
            The lightning module instance.
        checkpoint : dict
            The checkpoint dictionary to load the state.

        Returns
        -------
        None
        """
        if "fidkid_logger" in checkpoint:
            self.val_epoch_step_count = checkpoint["fidkid_logger"]["val_epoch_step_count"]
