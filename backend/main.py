from __future__ import annotations

import cv2
import numpy as np

# Audio is currently disabled for sign language vision development
# from audio import start_audio_thread
from vision import get_latest_event, get_latest_frame_bytes, start_vision_stream, stop_vision_stream


def _placeholder_frame(message: str = "Starting camera..."):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> int:
    # Audio thread disabled for now to focus purely on vision
    # if not start_audio_thread():
    #     print("[AUDIO]: Vision mode will continue without live speech translation.")

    if not start_vision_stream():
        print("[VISION]: Failed to start the local camera stream.")
        return 1

    last_frame = None

    try:
        while True:
            frame_bytes = get_latest_frame_bytes()
            if frame_bytes:
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                decoded_frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if decoded_frame is not None:
                    last_frame = decoded_frame

            frame = last_frame if last_frame is not None else _placeholder_frame()
            cv2.imshow("Lexis - Step 2", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return 0
    finally:
        stop_vision_stream()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
