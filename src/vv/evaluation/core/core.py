import typing as Typing


class CheckpointInfo(Typing.NamedTuple):
    ckpt_path: Typing.Union[str, None]
    config_path: Typing.Union[str, None]
