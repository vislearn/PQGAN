import inspect
import re
import typing as Typing
from datetime import datetime

import torch
import yaml  # type: ignore
from lightning.pytorch import Trainer
from lightning.pytorch.profilers import SimpleProfiler

import vv.evaluation.core.definitions as eval_def
from vv.data.core.data_class import CoreDataModule
from vv.evaluation.core.core import CheckpointInfo
from vv.evaluation.core.result_data import ExperimentResult, MetricResult
from vv.evaluation.runtime.eval_logger import InMemoryLogger
from vv.logger.codebook_analyser import CodebookAnalyzer
from vv.logger.metrics_logger import MetricsLogger
from vv.metrics.configurations import CMMDConfig, FIDConfig, ISConfig, KIDConfig, LPIPSConfig, PSNRConfig


class ModelEvaluator:
    def __init__(self,
                 eval_config_path: str) -> None:
        # Load saved configuration
        self.eval_config = self._load_config_file(eval_config_path)
        self.data_config = self.eval_config["data"]
        self.metrics_logger_config = self.eval_config["metrics_logger"]
        self.trainer_args = self.eval_config["trainer"]
        self.codebook_utilization_analyzer_args = self.eval_config.get("codebook_analyzer", None)
        self.overwrite_model_configs = self.eval_config.get("overwrite_model_configs", None)

        # Speed test
        self.run_speed_test = self.eval_config.get("run_speed_test", False)
        self.speed_test_only = self.eval_config.get("speed_test_only", False)
        self.speed_test_sample_size = self.eval_config.get("speed_test_sample_size", 100)

        if self.trainer_args is None:
            self.trainer_args = {}

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Init interface logger
        self.logger = InMemoryLogger()

        # Create data module
        data_module: CoreDataModule = self._load_class(self.data_config)

        # Metrics logger
        self.metrics_module: MetricsLogger = self._load_class_metrics(self.metrics_logger_config)
        renormalizer = data_module.get_renormalizer(target_min=0.0, target_max=1.0)
        self.metrics_module.renormalizer = renormalizer

        # Codebook utilization logger
        if self.codebook_utilization_analyzer_args is not None:
            self.codebook_utilization_analyzer: CodebookAnalyzer = self._load_class(
                config=self.codebook_utilization_analyzer_args
            )
        else:
            self.codebook_utilization_analyzer = None

        # Move metrics to GPU if available
        self.metrics_module.metrics = self.metrics_module.metrics.to(device)

        # Callback list
        self.callbacks = [self.metrics_module]
        if self.codebook_utilization_analyzer is not None:
            self.callbacks.append(self.codebook_utilization_analyzer)

    def evaluate_model(self,
                       ckpt_info: CheckpointInfo,
                       batch_size_reduction: float = 1.0) -> ExperimentResult:
        """
        Evaluates a model

        Parameters:
        -----------
        ckpt_info: CheckpointInfo
            The checkpoint information containing the path to the model checkpoint and configuration.
        batch_size_reduction: float
            The factor by which to reduce the batch size for evaluation.

        Returns:
        -----------
        ExperimentResult: The result of the evaluation containing metrics and other information.
        """
        # Run speed test if required
        if self.run_speed_test:
            speed_test_log = self._execute_speed_test(ckpt_info, disable_quantization=False, batch_size_reduction=batch_size_reduction)
            speed_test_log_no_quant = self._execute_speed_test(ckpt_info, disable_quantization=True, batch_size_reduction=batch_size_reduction)

        # Run standard metrics evaluation
        if not self.speed_test_only:
            metrics_log = self._execute_metrics_test(ckpt_info, batch_size_reduction)
        else:
            metrics_log = {eval_def.TIMESTAMP_KEY_NAME: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           eval_def.METRICS_KEY_NAME: []}

        result: ExperimentResult = ExperimentResult()
        result.experiment_id = ckpt_info.ckpt_path
        result.timestamp = metrics_log[eval_def.TIMESTAMP_KEY_NAME]
        result.step = self._extract_step_from_ckpt_path(ckpt_info.ckpt_path)
        result.metrics = metrics_log[eval_def.METRICS_KEY_NAME]
        if self.run_speed_test:
            result.metrics += [speed_test_log[eval_def.SPEED_TEST_KEY_NAME]]
            result.metrics += [speed_test_log_no_quant[eval_def.SPEED_TEST_KEY_NAME]]
        if self.codebook_utilization_analyzer is not None:
            result.codebook_statistics = self.logger.get_histogram()
            self.logger.reset_histogram()

        return result

    def _execute_metrics_test(self,
                              ckpt_info: CheckpointInfo,
                              batch_size_reduction: float = 1.0) -> dict:
        """
        Executes the evaluation of standard reconstruction metrics.

        Parameters:
        -----------
        ckpt_info: CheckpointInfo
            The checkpoint information containing the path to the model checkpoint and configuration.
        batch_size_reduction: float
            The factor by which to reduce the batch size for evaluation.

        Returns:
        -----------
        dict: The logged metrics from the evaluation.
        """
        # Create data module
        data_config_reduced = self._reduce_batch_size(self.data_config, batch_size_reduction)
        data_module: CoreDataModule = self._load_class(data_config_reduced)
        data_module.prepare_data()
        data_module.setup("validate")

        # Load saved configuration
        model_config_file = self._load_config_file(ckpt_info.config_path)
        model_cfg = model_config_file["model"]
        model_cfg = self._overwrite_model_configs(model_cfg, self.overwrite_model_configs)

        # Create model
        model = self._load_class(model_cfg, ckpt_path=ckpt_info.ckpt_path)
        model.log = lambda *args, **kwargs: None

        trainer = Trainer(
            logger=self.logger,
            callbacks=self.callbacks,
            **self.trainer_args
        )

        # Run validation
        trainer.validate(model, datamodule=data_module, verbose=False)
        self.metrics_module.metrics.reset()

        # Extract results
        metrics_log = self.logger.get_last_log()

        return metrics_log

    def _execute_speed_test(self,
                            ckpt_info: CheckpointInfo,
                            disable_quantization: bool = False,
                            batch_size_reduction: float = 1.0) -> dict:
        """
        Executes the model inference speed test.

        Parameters:
        -----------
        ckpt_info: CheckpointInfo
            The checkpoint information containing the path to the model checkpoint and configuration.
        disable_quantization: bool
            Whether to disable quantization during evaluation.
        batch_size_reduction: float
            The factor by which to reduce the batch size for evaluation.

        Returns:
        -----------
        dict: A dictionary containing the speed test results.
        """
        # Create data module
        data_config_reduced = self._reduce_batch_size(self.data_config, batch_size_reduction)
        data_module: CoreDataModule = self._load_class(data_config_reduced)
        data_module.prepare_data()
        data_module.setup("validate")

        # Load saved configuration
        model_config_file = self._load_config_file(ckpt_info.config_path)
        model_cfg = model_config_file["model"]
        if disable_quantization:
            overwrite_model_configs = self.overwrite_model_configs or {}
            if "init_args" not in overwrite_model_configs:
                overwrite_model_configs["init_args"] = {}
            overwrite_model_configs["init_args"]["disable_quantization"] = True
        else:
            overwrite_model_configs = self.overwrite_model_configs or {}

        model_cfg = self._overwrite_model_configs(model_cfg, overwrite_model_configs)

        # Create model
        model = self._load_class(model_cfg, ckpt_path=ckpt_info.ckpt_path)
        model.log = lambda *args, **kwargs: None

        # Use a profiler to measure inference time
        trainer_args = self.trainer_args.copy()
        profiler = SimpleProfiler()
        trainer_args['profiler'] = profiler
        trainer_args['limit_val_batches'] = self.speed_test_sample_size

        trainer = Trainer(
            logger=self.logger,
            **trainer_args
        )

        # Run validation to trigger inference
        trainer.validate(model, datamodule=data_module, verbose=False)

        # Extract profiler results
        profiler_results = trainer.profiler._make_report()
        validation_step_time = 0.0
        validation_time_total = 0.0
        for action, mean_duration, total_duration in profiler_results:
            if ".validation_step" in action:
                validation_step_time = mean_duration
                validation_time_total = total_duration
                break

        if disable_quantization:
            name = "samples_per_second_no_quantization"
        else:
            name = "samples_per_second"

        # Create a log entry for the speed test
        speed_test_log: dict = {
            eval_def.SPEED_TEST_KEY_NAME: MetricResult(
                name=name,
                value=validation_step_time,
                details={
                    "total_time_seconds": validation_time_total
                }
            )
        }

        return speed_test_log

    def _load_config_file(self,
                          config_path: str) -> Typing.Dict[str, Typing.Any]:
        """
        Loads the model configuration file.

        Parameters:
        -----------
        config_path: str
            The path to the configuration file.

        Returns:
        -----------
        dict: The loaded configuration as a dictionary.
        """
        with open(config_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config

    def _import_class(self,
                      class_import_path: str) -> Typing.Any:
        """
        Imports a class from a given module path.

        Parameters:
        -----------
        class_import_path: str
            The path to the class in the format 'module.submodule.ClassName'.

        Returns:
        -----------
        class: The imported class.
        """
        module_path, class_name = class_import_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def _validate_config_parameters(self,
                                    init_args: Typing.Dict,
                                    module: Typing.Any) -> None:
        """
        Validates the configuration parameters.

        Parameters:
        -----------
        config: dict
            The configuration dictionary to validate.

        Returns:
        -----------
        None
        """
        # Validate init_args against the model class constructor
        model_constructor = inspect.signature(module.__init__)
        valid_args = model_constructor.parameters.keys()

        # Check for unexpected arguments (not in the model constructor, but in the config)
        unexpected_args = [arg for arg in init_args if arg not in valid_args]
        if unexpected_args:
            raise ValueError(
                f"Unexpected arguments in module configuration: {unexpected_args}. "
                f"Valid arguments are: {list(valid_args)}"
            )

        # Check for missing required arguments
        required_args = [
            name for name, param in model_constructor.parameters.items()
            if param.default == inspect.Parameter.empty and name != "self"
        ]
        missing_args = [arg for arg in required_args if arg not in init_args]
        if missing_args:
            raise ValueError(
                f"Missing required arguments in module configuration: {missing_args}. "
                f"Required arguments are: {required_args}"
            )

    def _reduce_batch_size(self,
                           data_config: Typing.Dict,
                           batch_size_reduction: float = 1.0) -> Typing.Dict:
        """
        Reduces the batch size in the data configuration.

        Parameters:
        -----------
        data_config: dict
            The data configuration dictionary.
        batch_size_reduction: float
            The factor by which to reduce the batch size.

        Returns:
        -----------
        dict: The modified data configuration with reduced batch size.
        """
        data_config = self.data_config.copy()
        data_config["init_args"]["batch_size"] = int(
            data_config["init_args"]["batch_size"] * batch_size_reduction
        )

        return data_config

    def _load_class(self,
                    config: Typing.Dict,
                    ckpt_path: Typing.Union[str, None] = None) -> Typing.Any:
        """
        Imports a class from a given module path.

        Parameters:
        -----------
        config: dict
            The configuration dictionary containing the class path and initialization arguments.

        Returns:
        -----------
        class: The imported class.
        """
        # Init module
        module = self._import_class(config["class_path"])
        # Verify model configuration
        self._validate_config_parameters(config["init_args"], module)
        # Load module
        if ckpt_path is not None:
            # In case of model checkpoint
            module = module.load_from_checkpoint(
                ckpt_path,
                strict=False,  # Allow missing keys for backward compatibility
                **config["init_args"]
            )
        else:
            # Every other case
            module = module(**config["init_args"])

        return module

    def _load_class_metrics(self,
                            config: Typing.Dict) -> Typing.Any:
        """
        Imports a class from a given module path.

        Parameters:
        -----------
        config: dict
            The configuration dictionary containing the class path and initialization arguments.

        Returns:
        -----------
        class: The imported class.
        """
        # Init module
        module = self._import_class(config["class_path"])
        # Verify model configuration
        self._validate_config_parameters(config["init_args"], module)

        # Convert list to tuple. Very annyoing
        if config["init_args"].get("psnr_config", {}).get("data_range") is not None:
            config["init_args"]["psnr_config"]["data_range"] = tuple(
                config["init_args"]["psnr_config"]["data_range"]
            )

        args = {
            "log_every_n_val_epoch": config["init_args"].get("log_every_n_val_epoch", 1),
            "fid_config": FIDConfig(**config["init_args"].get("fid_config", {})),
            "kid_config": KIDConfig(**config["init_args"].get("kid_config", {})),
            "is_config": ISConfig(**config["init_args"].get("is_config", {})),
            "lpips_config": LPIPSConfig(**config["init_args"].get("lpips_config", {})),
            "psnr_config": PSNRConfig(**config["init_args"].get("psnr_config", {})),
            "cmmd_config": CMMDConfig(**config["init_args"].get("cmmd_config", {})),
        }

        # Load module
        module = module(**args)

        return module

    def _overwrite_model_configs(self,
                                 model_cfg_old: Typing.Dict,
                                 model_cfg_overwrite: Typing.Dict) -> Typing.Dict:
        """
        Overwrite the original model configs with the new given ones.

        Parameters:
        -----------
        model_cfg: dict
            The model configuration dictionary.

        Returns:
        -----------
        None
        """
        if model_cfg_overwrite is None:
            return model_cfg_old

        # Merge the old and overwrite configs
        if "init_args" in model_cfg_overwrite and model_cfg_overwrite["init_args"]:
            for key, value in model_cfg_overwrite["init_args"].items():
                model_cfg_old["init_args"][key] = value

        return model_cfg_old

    def _extract_step_from_ckpt_path(self,
                                     ckpt_path: str) -> int:
        """
        Extracts the step number from a checkpoint path.

        Parameters:
        -----------
        ckpt_path: str
            The path to the checkpoint file.

        Returns:
        -----------
        int: The extracted step number.
        """
        match = re.search(r"step=(\d+)", ckpt_path)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Step number not found in checkpoint path: {ckpt_path}")
