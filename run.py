from tqdm import tqdm

from pipeline import (CaptureVideo,
                      ScaleVideo,
                      DefineROI,
                      TrackRedBalls,
                      TrackBlueBalls,
                      TrackWhiteBall,
                      MergeRoi,
                      DisplayScore,
                      DisplayVideo)

if __name__ == "__main__":
    capture_video = CaptureVideo("videos/sinuquinha.mp4")
    display_video = DisplayVideo("image")
    define_roi = DefineROI((88, 413), (100, 660))
    scale_video = ScaleVideo(.4)
    track_red_balls = TrackRedBalls()
    track_blue_balls = TrackBlueBalls()
    track_white_ball = TrackWhiteBall()
    display_score = DisplayScore("Jogador 1", "Jogador 2", 4)
    merge_roi = MergeRoi()

    pipeline = (capture_video | scale_video | define_roi |
                track_red_balls | track_blue_balls | track_white_ball | merge_roi | display_score | display_video)

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
