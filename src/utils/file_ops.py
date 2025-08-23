"""
File operation utilities for the crypto algorithmic trading system.

Provides safe and efficient file operations:
- Directory creation and management
- Safe file saving with atomic operations
- File validation and integrity checks
- Backup and recovery operations
"""

import hashlib
import json
import logging
import pickle
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FileOperationError(Exception):
    """Custom exception for file operation errors."""
    pass


def ensure_directory(path: str | Path, create_parents: bool = True) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path
        create_parents: Whether to create parent directories

    Returns:
        Path object for the directory

    Raises:
        FileOperationError: If directory creation fails
    """
    try:
        path_obj = Path(path)
        if create_parents:
            path_obj.mkdir(parents=True, exist_ok=True)
        else:
            path_obj.mkdir(exist_ok=True)

        logger.debug(f"Directory ensured: {path_obj}")
        return path_obj

    except Exception as e:
        error_msg = f"Failed to create directory {path}: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg) from e


def safe_save_file(
    data: Any,
    filepath: str | Path,
    file_format: str = 'auto',
    backup: bool = True,
    compression: str = None,
    **kwargs
) -> Path:
    """
    Safely save data to a file using atomic operations.

    Args:
        data: Data to save
        filepath: Target file path
        file_format: File format ('auto', 'csv', 'parquet', 'json', 'yaml', 'pickle')
        backup: Whether to create a backup of existing file
        compression: Compression method for supported formats
        **kwargs: Additional arguments for the save method

    Returns:
        Path object for the saved file

    Raises:
        FileOperationError: If file saving fails
    """
    try:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)

        # Create backup if requested and file exists
        if backup and filepath.exists():
            backup_path = filepath.with_suffix(f"{filepath.suffix}.backup")
            shutil.copy2(filepath, backup_path)
            logger.debug(f"Created backup: {backup_path}")

        # Determine file format if auto
        if file_format == 'auto':
            file_format = filepath.suffix.lstrip('.')

        # Save to temporary file first
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=filepath.suffix,
            dir=filepath.parent
        )
        temp_path = Path(temp_file.name)

        try:
            # Save data based on format
            if file_format in ['csv', 'txt']:
                if isinstance(data, pd.DataFrame):
                    data.to_csv(temp_path, index=False, compression=compression, **kwargs)
                else:
                    raise ValueError(f"Cannot save {type(data)} to CSV format")

            elif file_format == 'parquet':
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(temp_path, compression=compression, **kwargs)
                else:
                    raise ValueError(f"Cannot save {type(data)} to Parquet format")

            elif file_format == 'json':
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str, **kwargs)

            elif file_format == 'yaml':
                with open(temp_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, **kwargs)

            elif file_format == 'pickle':
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f, **kwargs)

            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            # Atomic move to final location
            shutil.move(str(temp_path), str(filepath))

            logger.info(f"File saved successfully: {filepath}")
            return filepath

        finally:
            # Clean up temporary file if it still exists
            if temp_path.exists():
                temp_path.unlink()

    except Exception as e:
        error_msg = f"Failed to save file {filepath}: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg) from e


def safe_load_file(
    filepath: str | Path,
    file_format: str = 'auto',
    fallback_format: str = None
) -> Any:
    """
    Safely load data from a file.

    Args:
        filepath: File path to load
        file_format: File format ('auto', 'csv', 'parquet', 'json', 'yaml', 'pickle')
        fallback_format: Fallback format if auto-detection fails

    Returns:
        Loaded data

    Raises:
        FileOperationError: If file loading fails
    """
    try:
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileOperationError(f"File does not exist: {filepath}")

        # Determine file format if auto
        if file_format == 'auto':
            file_format = filepath.suffix.lstrip('.')

        # Load data based on format
        if file_format in ['csv', 'txt']:
            data = pd.read_csv(filepath)

        elif file_format == 'parquet':
            data = pd.read_parquet(filepath)

        elif file_format == 'json':
            with open(filepath) as f:
                data = json.load(f)

        elif file_format == 'yaml':
            with open(filepath) as f:
                data = yaml.safe_load(f)

        elif file_format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

        else:
            # Try fallback format
            if fallback_format:
                logger.warning(f"Unknown format {file_format}, trying {fallback_format}")
                return safe_load_file(filepath, fallback_format)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

        logger.debug(f"File loaded successfully: {filepath}")
        return data

    except Exception as e:
        error_msg = f"Failed to load file {filepath}: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg) from e


def calculate_file_hash(filepath: str | Path, algorithm: str = 'sha256') -> str:
    """
    Calculate file hash for integrity verification.

    Args:
        filepath: File path
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')

    Returns:
        Hexadecimal hash string

    Raises:
        FileOperationError: If hash calculation fails
    """
    try:
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileOperationError(f"File does not exist: {filepath}")

        hash_obj = hashlib.new(algorithm)

        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)

        hash_value = hash_obj.hexdigest()
        logger.debug(f"File hash calculated: {filepath} -> {algorithm}:{hash_value}")
        return hash_value

    except Exception as e:
        error_msg = f"Failed to calculate hash for {filepath}: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg) from e


def verify_file_integrity(
    filepath: str | Path,
    expected_hash: str,
    algorithm: str = 'sha256'
) -> bool:
    """
    Verify file integrity using hash comparison.

    Args:
        filepath: File path to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm used

    Returns:
        True if hash matches, False otherwise
    """
    try:
        actual_hash = calculate_file_hash(filepath, algorithm)
        is_valid = actual_hash == expected_hash

        if is_valid:
            logger.info(f"File integrity verified: {filepath}")
        else:
            logger.warning(f"File integrity check failed: {filepath}")
            logger.warning(f"Expected: {expected_hash}, Got: {actual_hash}")

        return is_valid

    except Exception as e:
        logger.error(f"Failed to verify file integrity: {filepath} - {e}")
        return False


def cleanup_old_files(
    directory: str | Path,
    pattern: str = "*",
    max_age_days: int = 30,
    dry_run: bool = False
) -> dict[str, Any]:
    """
    Clean up old files in a directory.

    Args:
        directory: Directory to clean
        pattern: File pattern to match
        max_age_days: Maximum age of files to keep
        dry_run: If True, only report what would be deleted

    Returns:
        Summary of cleanup operation
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            return {'message': 'Directory does not exist', 'deleted': 0}

        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        deleted_files = []
        total_size = 0

        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_age = file_path.stat().st_mtime
                if file_age < cutoff_time:
                    file_size = file_path.stat().st_size
                    deleted_files.append({
                        'path': str(file_path),
                        'size': file_size,
                        'age_days': (datetime.now().timestamp() - file_age) / (24 * 3600)
                    })
                    total_size += file_size

                    if not dry_run:
                        file_path.unlink()
                        logger.debug(f"Deleted old file: {file_path}")

        summary = {
            'directory': str(directory),
            'pattern': pattern,
            'max_age_days': max_age_days,
            'dry_run': dry_run,
            'deleted_count': len(deleted_files),
            'deleted_size_bytes': total_size,
            'deleted_files': deleted_files
        }

        if dry_run:
            logger.info(f"Dry run cleanup: would delete {len(deleted_files)} files")
        else:
            logger.info(f"Cleanup completed: deleted {len(deleted_files)} files")

        return summary

    except Exception as e:
        error_msg = f"Failed to cleanup directory {directory}: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg) from e


def get_directory_info(directory: str | Path) -> dict[str, Any]:
    """
    Get comprehensive information about a directory.

    Args:
        directory: Directory path

    Returns:
        Directory information dictionary
    """
    try:
        directory = Path(directory)

        if not directory.exists():
            return {'error': 'Directory does not exist'}

        if not directory.is_dir():
            return {'error': 'Path is not a directory'}

        total_files = 0
        total_dirs = 0
        total_size = 0
        file_types = {}

        for item in directory.rglob('*'):
            if item.is_file():
                total_files += 1
                file_size = item.stat().st_size
                total_size += file_size

                # Count file types
                suffix = item.suffix.lower()
                file_types[suffix] = file_types.get(suffix, 0) + 1

            elif item.is_dir():
                total_dirs += 1

        info = {
            'path': str(directory),
            'total_files': total_files,
            'total_directories': total_dirs,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_types': file_types,
            'last_modified': datetime.fromtimestamp(directory.stat().st_mtime).isoformat()
        }

        return info

    except Exception as e:
        error_msg = f"Failed to get directory info for {directory}: {e}"
        logger.error(error_msg)
        return {'error': error_msg}
