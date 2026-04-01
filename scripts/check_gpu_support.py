"""
Quick GPU capability check for the Smart Attendance project.
"""
from __future__ import annotations

import sys


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def check_tensorflow() -> bool:
    print_header("TensorFlow")
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"TensorFlow import failed: {exc}")
        return False

    print(f"Version: {tf.__version__}")
    gpu_devices = tf.config.list_physical_devices("GPU")
    print(f"GPUs seen by TensorFlow: {gpu_devices}")

    if gpu_devices:
        print("Status: GPU using")
        return True

    print("Status: CPU only")
    return False


def check_dlib() -> bool:
    print_header("dlib / face_recognition")
    try:
        import dlib
    except Exception as exc:
        print(f"dlib import failed: {exc}")
        return False

    cuda_enabled = bool(getattr(dlib, "DLIB_USE_CUDA", False))
    print(f"dlib version: {getattr(dlib, '__version__', 'unknown')}")
    print(f"DLIB_USE_CUDA: {cuda_enabled}")

    try:
        device_count = dlib.cuda.get_num_devices()
    except Exception as exc:
        print(f"CUDA device query failed: {exc}")
        device_count = 0

    print(f"dlib CUDA device count: {device_count}")

    if cuda_enabled and device_count > 0:
        print("Status: GPU using")
        return True

    print("Status: CPU only")
    return False


def check_mediapipe_note() -> None:
    print_header("MediaPipe")
    print("This project's current MediaPipe Python path is expected to run on CPU.")
    print("So even with TensorFlow + dlib on GPU, pose extraction may still stay CPU-bound.")


def main() -> int:
    tf_gpu = check_tensorflow()
    dlib_gpu = check_dlib()
    check_mediapipe_note()

    print_header("Summary")
    print(f"TensorFlow: {'GPU using' if tf_gpu else 'CPU only'}")
    print(f"dlib/face_recognition: {'GPU using' if dlib_gpu else 'CPU only'}")

    if tf_gpu and dlib_gpu:
        print("Overall: The main GPU-accelerated parts are ready.")
        return 0

    print("Overall: Full GPU acceleration is NOT ready yet.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
