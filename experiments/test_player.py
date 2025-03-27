import ffmpeg
import cv2
import numpy as np
import threading
import os

def play_video_with_ffmpeg_python(video_path):
    """
    Play a video using ffmpeg-python to extract frames and OpenCV to display them.
    """
    # Probe the video to get its width and height
    probe = ffmpeg.probe(video_path)
    video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    # Use ffmpeg-python to read raw video frames
    process = (
        ffmpeg
        .input(video_path, re=None, hwaccel='d3d12va', hwaccel_device='1')  # Use Intel iGPU (device 0)
        # .input(video_path, re=None)  # Use CUDA for hardware acceleration if available
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run_async(pipe_stdout=True, pipe_stderr=True)  # Run asynchronously
    )

    # Function to consume and print stderr
    def consume_stderr(pipe):
        # for line in iter(pipe.readline, b''):
        #     print(line.decode('utf-8').strip())  # Print FFmpeg output line by line
        pipe.close()

    # Start a thread to consume stderr
    stderr_thread = threading.Thread(target=consume_stderr, args=(process.stderr,))
    stderr_thread.start()

    try:
        while True:
            # Read raw frame data from FFmpeg's stdout
            frame_size = width * height * 3  # 3 bytes per pixel for bgr24
            in_bytes = process.stdout.read(frame_size)

            if not in_bytes:
                break  # End of video

            # Convert raw frame to a NumPy array
            frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))

            # Display the frame using OpenCV
            cv2.imshow("Video", frame)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Cleaning up... 1")
        process.stdout.close()
        print("Cleaning up... 2")
        process.wait()
        print("Cleaning up... 3")
        stderr_thread.join()  # Ensure the stderr thread finishes
        cv2.destroyAllWindows()

# Example usage
video_path = "C:/Developer/DepthAI/DepthPose/recordings/20250326-213136_4_C_L_R/0_C_000.mp4"  # Replace with your video file path
play_video_with_ffmpeg_python(video_path)