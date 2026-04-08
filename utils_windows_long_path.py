
import os


def maybe_windows_long_path(path):
    """
    Work around Windows MAX_PATH limitations by using the extended-length path prefix.

    On some Windows setups, operations like open()/os.remove()/shutil.copy2 may fail with
    FileNotFoundError when the *absolute* path exceeds ~260 characters. Prefixing with
    '\\\\?\\' enables long path support for Win32 APIs.
    """
    try:
        path_str = os.fspath(path)
    except TypeError:
        return path

    if os.name != "nt" or not path_str:
        return path_str

    # 已经是长路径前缀
    if path_str.startswith("\\\\?\\"):
        return path_str

    abs_path = os.path.abspath(path_str)
    abs_path = os.path.normpath(abs_path)

    # UNC 网络路径
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    # 本地盘符路径
    return "\\\\?\\" + abs_path


def safe_open(file_path, *args, **kwargs):
    return open(maybe_windows_long_path(file_path), *args, **kwargs)