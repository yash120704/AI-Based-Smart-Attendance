"""
One-time migration of local face images and behavior models to Cloudinary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import KNOWN_FACES_DIR, MODELS_DIR
from utils.cloud_storage import upload_face_image, upload_model_file

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_PATTERNS = ("*.h5", "*.pkl")


def migrate_faces():
    uploaded = []
    skipped = []

    for person_dir in sorted(KNOWN_FACES_DIR.glob("*")):
        if not person_dir.is_dir():
            continue

        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped.append(image_path)
                continue

            url = upload_face_image(person_dir.name, image_path.read_bytes())
            uploaded.append((person_dir.name, image_path.name, url))

    return uploaded, skipped


def migrate_models():
    uploaded = []
    seen = set()

    for pattern in MODEL_PATTERNS:
        for model_path in sorted(MODELS_DIR.glob(pattern)):
            if model_path in seen:
                continue
            seen.add(model_path)
            url = upload_model_file(model_path.name, model_path.read_bytes())
            uploaded.append((model_path.name, url))

    return uploaded


def main():
    print("Migrating Smart Attendance local assets to Cloudinary")
    print(f"Faces directory: {KNOWN_FACES_DIR}")
    print(f"Models directory: {MODELS_DIR}")

    face_uploads, skipped_faces = migrate_faces()
    model_uploads = migrate_models()

    print("\nFace images uploaded:")
    if face_uploads:
        for person_name, filename, url in face_uploads:
            print(f"  {person_name}/{filename} -> {url}")
    else:
        print("  none")

    if skipped_faces:
        print("\nFace files skipped:")
        for path in skipped_faces:
            print(f"  {path}")

    print("\nModel files uploaded:")
    if model_uploads:
        for filename, url in model_uploads:
            print(f"  {filename} -> {url}")
    else:
        print("  none")

    print("\nSummary:")
    print(f"  Face images uploaded: {len(face_uploads)}")
    print(f"  Face files skipped: {len(skipped_faces)}")
    print(f"  Model files uploaded: {len(model_uploads)}")


if __name__ == "__main__":
    main()
