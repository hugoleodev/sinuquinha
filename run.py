from tqdm import tqdm

from pipeline.capture_video import CaptureVideo
from pipeline.display_video import DisplayVideo
from pipeline.define_roi import DefineROI
from pipeline.scale_video import ScaleVideo

if __name__ == "__main__":
    capture_video = CaptureVideo("videos/sinuquinha.mp4")
    display_video = DisplayVideo("image")
    define_roi = DefineROI((88, 413), (100, 660))
    scale_video = ScaleVideo(.4)

    pipeline = (capture_video | scale_video | define_roi | display_video)

    progress = tqdm(
        total=capture_video.frame_count if capture_video.frame_count > 0 else None)

    try:
        for _ in pipeline:
            progress.update(1)
    except StopIteration:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        progress.close()
