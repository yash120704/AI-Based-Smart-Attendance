"""
Cloudinary storage helpers for cloud deployments.
"""
import io
import os
import uuid
from pathlib import Path
from typing import Dict, List

FACES_FOLDER = "smart-attendance/faces"
MODELS_FOLDER = "smart-attendance/models"


def is_cloud_storage_enabled() -> bool:
    """
    Return True when Cloudinary credentials are present.
    """
    return all(
        os.environ.get(name)
        for name in (
            "CLOUDINARY_CLOUD_NAME",
            "CLOUDINARY_API_KEY",
            "CLOUDINARY_API_SECRET",
        )
    )


def _cloudinary_modules():
    if not is_cloud_storage_enabled():
        raise RuntimeError("Cloudinary credentials are not configured")

    try:
        import cloudinary
        import cloudinary.api
        import cloudinary.uploader
    except ImportError as exc:
        raise RuntimeError("cloudinary is required when CLOUDINARY_* env vars are set") from exc

    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
        secure=True,
    )
    return cloudinary.api, cloudinary.uploader


def _bytes_file(file_bytes: bytes, filename: str) -> io.BytesIO:
    buffer = io.BytesIO(file_bytes)
    buffer.name = filename
    buffer.seek(0)
    return buffer


def upload_face_image(name, image_bytes) -> str:
    """
    Upload one face image for a person.

    Args:
        name: Person name
        image_bytes: Encoded image bytes

    Returns:
        str: Secure Cloudinary URL
    """
    _, uploader = _cloudinary_modules()
    clean_name = str(name).replace(" ", "_")
    public_id = f"{FACES_FOLDER}/{clean_name}/face_{uuid.uuid4().hex}"
    result = uploader.upload(
        _bytes_file(image_bytes, f"{clean_name}.jpg"),
        public_id=public_id,
        resource_type="image",
        overwrite=True,
    )
    return result["secure_url"]


def upload_model_file(filename, file_bytes) -> str:
    """
    Upload one model artifact.

    Args:
        filename: Model artifact filename
        file_bytes: File contents

    Returns:
        str: Secure Cloudinary URL
    """
    _, uploader = _cloudinary_modules()
    file_name = Path(filename).name
    public_id = f"{MODELS_FOLDER}/{file_name}"
    result = uploader.upload(
        _bytes_file(file_bytes, file_name),
        public_id=public_id,
        resource_type="raw",
        overwrite=True,
    )
    return result["secure_url"]


def download_model_file(filename) -> bytes:
    """
    Download one model artifact by filename.

    Args:
        filename: Model artifact filename

    Returns:
        bytes: Downloaded model artifact contents
    """
    api, _ = _cloudinary_modules()
    file_name = Path(filename).name
    public_id = f"{MODELS_FOLDER}/{file_name}"

    try:
        resource = api.resource(public_id, resource_type="raw")
    except Exception:
        resource = api.resource(public_id.removesuffix(Path(file_name).suffix), resource_type="raw")

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required to download Cloudinary model files") from exc

    response = requests.get(resource["secure_url"], timeout=60)
    response.raise_for_status()
    return response.content


def _list_resources(prefix: str, resource_type: str) -> List[dict]:
    api, _ = _cloudinary_modules()
    resources = []
    next_cursor = None

    while True:
        kwargs = {
            "type": "upload",
            "prefix": prefix,
            "resource_type": resource_type,
            "max_results": 500,
        }
        if next_cursor:
            kwargs["next_cursor"] = next_cursor

        response = api.resources(**kwargs)
        resources.extend(response.get("resources", []))
        next_cursor = response.get("next_cursor")
        if not next_cursor:
            break

    return sorted(resources, key=lambda item: item.get("public_id", ""))


def list_face_images(name) -> List[str]:
    """
    List all face image URLs for a person.

    Args:
        name: Person name

    Returns:
        list[str]: Secure image URLs
    """
    clean_name = str(name).replace(" ", "_")
    prefix = f"{FACES_FOLDER}/{clean_name}/"
    return [resource["secure_url"] for resource in _list_resources(prefix, "image")]


def list_all_face_images() -> Dict[str, List[str]]:
    """
    List every Cloudinary face image grouped by person name.
    """
    grouped: Dict[str, List[str]] = {}
    for resource in _list_resources(f"{FACES_FOLDER}/", "image"):
        public_id = resource.get("public_id", "")
        parts = public_id.split("/")
        if len(parts) < 4:
            continue
        person_name = parts[2]
        grouped.setdefault(person_name, []).append(resource["secure_url"])
    return grouped
