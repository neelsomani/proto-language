"""
Storage module for large file content.

This module provides file storage backends for externalizing large file content
(PDB, CIF, HMM, etc.) from sequence metadata. It supports local filesystem storage
for development and Google Cloud Storage for production.

Configuration via environment variables:
    FILE_STORE_TYPE: "local" (default) or "gcs"
    FILE_STORE_PATH: Base path for local storage (default: ./file_store)
    GCS_FILE_BUCKET: GCS bucket name (required if FILE_STORE_TYPE=gcs)

Example usage in constraints:
    >>> from proto_language.storage import store_file, FileType
    >>> seq._metadata["pdb_output"] = store_file(structure.structure_pdb, FileType.PDB)

Example usage to retrieve content:
    >>> from proto_language.storage import get_file_content
    >>> pdb_content = get_file_content(seq._metadata["pdb_output"])
"""

from proto_language.storage.helpers import (
    get_file_content,
    get_file_content_bytes,
    is_file_reference,
    store_file,
)
from proto_language.storage.models import FILE_REF_MARKER, FileReference, FileType
from proto_language.storage.store import (
    FileStore,
    GCSFileStore,
    LocalFileStore,
    get_file_store,
    reset_file_store,
)

__all__ = [
    # Models
    "FileType",
    "FileReference",
    "FILE_REF_MARKER",
    # Stores
    "FileStore",
    "LocalFileStore",
    "GCSFileStore",
    "get_file_store",
    "reset_file_store",
    # Helpers
    "store_file",
    "get_file_content",
    "get_file_content_bytes",
    "is_file_reference",
]
