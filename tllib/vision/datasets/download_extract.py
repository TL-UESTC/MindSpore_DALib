import urllib.request
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Callable, IO
import gzip
import lzma
import pathlib
import os
import zipfile
import tarfile
import contextlib

USER_AGENT = "MindSpore/2.2"
_COMPRESSED_FILE_OPENERS: Dict[str, Callable[..., IO]] = {".gz": gzip.open, ".xz": lzma.open}
_FILE_TYPE_ALIASES: Dict[str, Tuple[Optional[str], Optional[str]]] = {".tgz": (".tar", ".gz")}
def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)

def download_url(url: str, root: str, filename: Optional[str] = None):
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    os.makedirs(root, exist_ok=True)
    try:
        print('Downloading ' + url + ' to ' + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                    ' Downloading ' + url + ' to ' + fpath)
            _urlretrieve(url, fpath)
        else:
            raise e

def _verify_archive_type(archive_type: str) -> None:
    if archive_type not in _ARCHIVE_EXTRACTORS.keys():
        valid_types = "', '".join(_ARCHIVE_EXTRACTORS.keys())
        raise RuntimeError(f"Unknown archive type '{archive_type}'. Known archive types are '{valid_types}'.")


def _verify_compression(compression: str) -> None:
    if compression not in _COMPRESSED_FILE_OPENERS.keys():
        valid_types = "', '".join(_COMPRESSED_FILE_OPENERS.keys())
        raise RuntimeError(f"Unknown compression '{compression}'. Known compressions are '{valid_types}'.")

def _detect_file_type(file: str) -> Tuple[str, Optional[str], Optional[str]]:
    path = pathlib.Path(file)
    suffix = path.suffix
    suffixes = pathlib.Path(file).suffixes
    if not suffixes:
        raise RuntimeError(
            f"File '{file}' has no suffixes that could be used to detect the archive type and compression."
        )
    elif len(suffixes) > 2:
        raise RuntimeError(
            "Archive type and compression detection only works for 1 or 2 suffixes. " f"Got {len(suffixes)} instead."
        )
    elif len(suffixes) == 2:
        # if we have exactly two suffixes we assume the first one is the archive type and the second on is the
        # compression
        archive_type, compression = suffixes
        return "".join(suffixes), archive_type, compression

    # check if the suffix is a known alias
    with contextlib.suppress(KeyError):
        return (suffix, *_FILE_TYPE_ALIASES[suffix])

    # check if the suffix is an archive type
    with contextlib.suppress(RuntimeError):
        _verify_archive_type(suffix)
        return suffix, suffix, None

    # check if the suffix is a compression
    with contextlib.suppress(RuntimeError):
        _verify_compression(suffix)
        return suffix, None, suffix
    
    raise RuntimeError(f"Suffix '{suffix}' is neither recognized as archive type nor as compression.")

def _extract_tar(from_path: str, to_path: str, compression: Optional[str]) -> None:
    with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
        tar.extractall(to_path)


_ZIP_COMPRESSION_MAP: Dict[str, int] = {
    ".xz": zipfile.ZIP_LZMA,
}


def _extract_zip(from_path: str, to_path: str, compression: Optional[str]) -> None:
    with zipfile.ZipFile(
        from_path, "r", compression=_ZIP_COMPRESSION_MAP[compression] if compression else zipfile.ZIP_STORED
    ) as zip:
        zip.extractall(to_path)


_ARCHIVE_EXTRACTORS: Dict[str, Callable[[str, str, Optional[str]], None]] = {
    ".tar": _extract_tar,
    ".zip": _extract_zip,
}

def _decompress(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> str:
    r"""Decompress a file.

    The compression is automatically detected from the file name.

    Args:
        from_path (str): Path to the file to be decompressed.
        to_path (str): Path to the decompressed file. If omitted, ``from_path`` without compression extension is used.
        remove_finished (bool): If ``True``, remove the file after the extraction.

    Returns:
        (str): Path to the decompressed file.
    """
    suffix, archive_type, compression = _detect_file_type(from_path)
    if not compression:
        raise RuntimeError(f"Couldn't detect a compression from suffix {suffix}.")

    if to_path is None:
        to_path = from_path.replace(suffix, archive_type if archive_type is not None else "")

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression]

    with compressed_file_opener(from_path, "rb") as rfh, open(to_path, "wb") as wfh:
        wfh.write(rfh.read())

    if remove_finished:
        os.remove(from_path)

    return to_path


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    suffix, archive_type, compression = _detect_file_type(from_path)

    if not archive_type:
        return _decompress(
            from_path,
            os.path.join(to_path, os.path.basename(from_path).replace(suffix, "")),
            remove_finished=remove_finished,
        )

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    extractor = _ARCHIVE_EXTRACTORS[archive_type]

    extractor(from_path, to_path, compression)

    return to_path

    

def download_and_extract_archive(url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    remove_finished: bool = False,
):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)
    
    download_url(url, download_root, filename)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)