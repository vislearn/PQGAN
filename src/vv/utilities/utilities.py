import argparse
import os


def get_dir(*args: str, create: bool = False) -> str:
    """
    Get the path to the data directory.

    Parameters
    ----------
    args : str
        The path to the data directory.
    create : bool, optional
        Whether to create the directory if it does not exist, by default False.

    Returns
    -------
    str
        The path to the data directory.
    """
    file_path = os.path.dirname(__file__)
    base_path = os.path.join(file_path, '..', '..', '..')
    final_path = os.path.abspath(os.path.join(base_path, *args))

    if create:
        os.makedirs(final_path, exist_ok=True)

    return final_path


def delete_file(*args: str) -> None:
    """
    Delete a file.

    Parameters
    ----------
    args : str
        The path to the file.
    """
    print(*args)
    file_path = os.path.join(get_dir(), *args)
    print(file_path)
    os.remove(file_path)


def str2bool(value: str) -> bool:
    """
    Converts a string to a boolean value.
    Accepts 'true', 'yes', '1' as True and 'false', 'no', '0' as False.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'yes', '1'):
        return True
    elif value.lower() in ('false', 'no', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: '{value}'")
