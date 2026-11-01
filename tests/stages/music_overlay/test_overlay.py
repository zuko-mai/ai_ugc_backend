import shutil
import subprocess
from pathlib import Path

import pytest

from app.stages.music_overlay.exceptions import InvalidSettingsError
from app.stages.music_overlay.overlay import (
    LIMITER_LINEAR,
    build_bed_filter,
    build_extract_filter,
    build_mix_filter,
    build_plan,
    overlay_music,
)
from app.stages.music_overlay.probe import probe_bytes, probe_file
from app.stages.music_overlay.schemas import OverlaySettings


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.skip(f"ffmpeg failed: {result.stderr[:300]}")


def _video(path: Path, *, seconds: float, with_audio: bool, size: str = "320x240") -> Path:
    _require_ffmpeg()
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=red:s={size}:d={seconds}:r=30",
    ]
    if with_audio:
        cmd.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={seconds}:sample_rate=44100",
                "-c:a",
                "aac",
                "-shortest",
            ]
        )
    else:
        cmd.append("-an")
    cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    _run(cmd)
    return path


def _wav(path: Path, *, seconds: float, rate: int = 48000, layout: str = "mono") -> Path:
    _require_ffmpeg()
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=1000:duration={seconds}:sample_rate={rate}",
            "-ac",
            "2" if layout == "stereo" else "1",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )
    return path


def _silence_then_tone(path: Path) -> Path:
    """1s of silence followed by 1s of tone. Used to prove the music offset."""
    _require_ffmpeg()
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1:sample_rate=48000",
            "-filter_complex",
            "[0:a][1:a]concat=n=2:v=0:a=1[a]",
            "-map",
            "[a]",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )
    return path


def _full_scale(path: Path, *, frequency: int) -> Path:
    _require_ffmpeg()
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"aevalsrc=sin({frequency}*2*PI*t):s=48000:d=1",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )
    return path


def _mean_max_db(path: Path, start: float, duration: float) -> tuple[float, float]:
    # Trim inside the filter. Output -ss on AAC still reports the earlier peak.
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            (
                f"atrim=start={start}:end={start + duration},"
                "asetpts=PTS-STARTPTS,volumedetect"
            ),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    mean = max_db = None
    for line in result.stderr.splitlines():
        if "mean_volume:" in line:
            mean = float(line.split("mean_volume:")[1].replace("dB", "").strip())
        if "max_volume:" in line:
            max_db = float(line.split("max_volume:")[1].replace("dB", "").strip())
    assert mean is not None and max_db is not None, result.stderr[-400:]
    return mean, max_db


def _peak_db(path: Path) -> float:
    """Highest per-channel peak. Do not downmix: stereo sum would exceed the limiter."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "astats=measure_overall=Peak_level",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    peaks = [
        float(line.split("Peak level dB:")[1].strip())
        for line in result.stderr.splitlines()
        if "Peak level dB:" in line
    ]
    assert peaks, result.stderr[-500:]
    return max(peaks)


def _video_md5(path: Path) -> str:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-f",
            "md5",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[:300]
    token = result.stdout.strip()
    assert token.startswith("MD5=")
    return token


def test_plan_matches_the_15s_example() -> None:
    plan = build_plan(
        video_duration=15,
        music_duration=120,
        settings=OverlaySettings(
            music_start_offset_seconds=35,
            video_start_seconds=0,
            music_volume=0.3,
            fade_out_seconds=1.5,
        ),
    )
    assert plan.music_start_offset_seconds == 35
    assert plan.video_start_seconds == 0
    assert plan.music_play_seconds == pytest.approx(15)
    assert plan.loop is False
    assert "music_trimmed_to_video" in plan.warnings


def test_offsets_are_independent() -> None:
    plan = build_plan(
        video_duration=15,
        music_duration=120,
        settings=OverlaySettings(
            music_start_offset_seconds=35,
            video_start_seconds=4,
            music_volume=0.3,
        ),
    )
    assert plan.music_start_offset_seconds == 35
    assert plan.video_start_seconds == 4
    assert plan.playback_window_seconds == pytest.approx(11)
    assert plan.music_play_seconds == pytest.approx(11)


def test_silence_and_trim_do_not_loop_when_music_is_shorter() -> None:
    silence = build_plan(
        video_duration=10,
        music_duration=3,
        settings=OverlaySettings(music_end_behavior="silence"),
    )
    trim = build_plan(
        video_duration=10,
        music_duration=3,
        settings=OverlaySettings(music_end_behavior="trim"),
    )
    assert silence.loop is False
    assert trim.loop is False
    assert silence.music_play_seconds == pytest.approx(3)
    assert trim.music_play_seconds == pytest.approx(3)
    assert "music_ended_before_video" in silence.warnings
    assert "music_shorter_than_video_no_loop" in trim.warnings


def test_loop_fills_the_remaining_video() -> None:
    plan = build_plan(
        video_duration=10,
        music_duration=3,
        settings=OverlaySettings(music_end_behavior="loop", music_start_offset_seconds=1),
    )
    assert plan.loop is True
    assert plan.available_music_seconds == pytest.approx(2)
    assert plan.music_play_seconds == pytest.approx(10)
    assert "music_looped_to_fill_video" in plan.warnings


def test_offset_past_music_duration_is_rejected() -> None:
    with pytest.raises(InvalidSettingsError) as exc:
        build_plan(
            video_duration=15,
            music_duration=120,
            settings=OverlaySettings(music_start_offset_seconds=120),
        )
    assert "music_start_offset_seconds" in exc.value.message


def test_video_start_past_video_duration_is_rejected() -> None:
    with pytest.raises(InvalidSettingsError):
        build_plan(
            video_duration=15,
            music_duration=120,
            settings=OverlaySettings(video_start_seconds=15),
        )


def test_fades_longer_than_playback_are_rejected() -> None:
    with pytest.raises(InvalidSettingsError):
        build_plan(
            video_duration=2,
            music_duration=30,
            settings=OverlaySettings(fade_in_seconds=1.5, fade_out_seconds=1.5),
        )


def test_filtergraph_uses_trim_fade_delay_mix_and_limiter() -> None:
    plan = build_plan(
        video_duration=15,
        music_duration=120,
        settings=OverlaySettings(
            music_start_offset_seconds=35,
            video_start_seconds=2,
            music_volume=0.3,
            fade_in_seconds=0.4,
            fade_out_seconds=1.5,
            original_audio_volume=0.8,
        ),
    )
    extract = build_extract_filter(plan.music_start_offset_seconds)
    bed = build_bed_filter(plan, 0.3)
    mixed = build_mix_filter(
        video_duration=15,
        video_start_seconds=plan.video_start_seconds,
        include_original=True,
        original_volume=0.8,
    )
    assert "atrim=start=35.000000" in extract
    assert "volume=0.300000" in bed
    assert "afade=t=in" in bed
    assert "afade=t=out" in bed
    assert "adelay=" in mixed
    assert "amix=" in mixed
    assert "normalize=0" in mixed
    assert f"alimiter=limit={LIMITER_LINEAR}" in mixed
    assert "volume=0.800000" in mixed

    music_only = build_mix_filter(
        video_duration=15,
        video_start_seconds=0,
        include_original=False,
        original_volume=1,
    )
    assert "amix=" not in music_only
    assert "adelay=" not in music_only
    assert "alimiter=" in music_only


def test_limiter_holds_full_scale_sum_under_the_ceiling(tmp_path: Path) -> None:
    _require_ffmpeg()
    left = _full_scale(tmp_path / "a.wav", frequency=440)
    right = _full_scale(tmp_path / "b.wav", frequency=660)
    graph = build_mix_filter(
        video_duration=1,
        video_start_seconds=0,
        include_original=True,
        original_volume=1,
    )
    limited = tmp_path / "limited.wav"
    unlimited = tmp_path / "unlimited.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            graph,
            "-map",
            "[aout]",
            "-c:a",
            "pcm_f32le",
            str(limited),
        ]
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
            "-map",
            "[a]",
            "-c:a",
            "pcm_f32le",
            str(unlimited),
        ]
    )
    assert _peak_db(unlimited) > 1.0
    assert _peak_db(limited) <= -0.2


@pytest.mark.asyncio
async def test_music_offset_skips_the_leading_silence(tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", seconds=1.0, with_audio=False)
    music = _silence_then_tone(tmp_path / "music.wav")
    video_probe = await probe_file(video)
    music_probe = await probe_file(music)
    settings = OverlaySettings(
        music_start_offset_seconds=1.0,
        video_start_seconds=0,
        music_volume=1,
        original_audio_enabled=False,
        music_end_behavior="trim",
    )
    result = await overlay_music(
        video_bytes=video.read_bytes(),
        music_bytes=music.read_bytes(),
        video_probe=video_probe,
        music_probe=music_probe,
        settings=settings,
        music_suffix=".wav",
    )
    out = tmp_path / "out.mp4"
    out.write_bytes(result.video_bytes)
    mean, _max_db = _mean_max_db(out, 0.05, 0.5)
    assert mean > -45

    early = OverlaySettings(
        music_start_offset_seconds=0,
        video_start_seconds=0,
        music_volume=1,
        original_audio_enabled=False,
    )
    quiet = await overlay_music(
        video_bytes=video.read_bytes(),
        music_bytes=music.read_bytes(),
        video_probe=video_probe,
        music_probe=music_probe,
        settings=early,
        music_suffix=".wav",
    )
    quiet_path = tmp_path / "quiet.mp4"
    quiet_path.write_bytes(quiet.video_bytes)
    quiet_mean, _quiet_max = _mean_max_db(quiet_path, 0.0, 0.45)
    assert quiet_mean < -60


@pytest.mark.asyncio
async def test_video_start_delays_music_independently_of_offset(tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", seconds=2.0, with_audio=False)
    music = _wav(tmp_path / "music.wav", seconds=2.0)
    video_probe = await probe_file(video)
    music_probe = await probe_file(music)
    result = await overlay_music(
        video_bytes=video.read_bytes(),
        music_bytes=music.read_bytes(),
        video_probe=video_probe,
        music_probe=music_probe,
        settings=OverlaySettings(
            music_start_offset_seconds=0,
            video_start_seconds=1.0,
            music_volume=1,
            original_audio_enabled=False,
        ),
        music_suffix=".wav",
    )
    out = tmp_path / "out.mp4"
    out.write_bytes(result.video_bytes)
    head, _ = _mean_max_db(out, 0.0, 0.45)
    tail, _ = _mean_max_db(out, 1.25, 0.4)
    assert head < -60
    assert tail > -45
    assert result.plan.music_start_offset_seconds == 0
    assert result.plan.video_start_seconds == 1


@pytest.mark.asyncio
async def test_loop_fills_tail_and_silence_does_not(tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", seconds=2.0, with_audio=False)
    music = _wav(tmp_path / "music.wav", seconds=0.5)
    video_probe = await probe_file(video)
    music_probe = await probe_file(music)

    async def _render(behavior: str) -> Path:
        result = await overlay_music(
            video_bytes=video.read_bytes(),
            music_bytes=music.read_bytes(),
            video_probe=video_probe,
            music_probe=music_probe,
            settings=OverlaySettings(
                music_volume=1,
                original_audio_enabled=False,
                music_end_behavior=behavior,  # type: ignore[arg-type]
            ),
            music_suffix=".wav",
        )
        path = tmp_path / f"{behavior}.mp4"
        path.write_bytes(result.video_bytes)
        return path

    looped = await _render("loop")
    silent = await _render("silence")
    loop_mean, _ = _mean_max_db(looped, 1.3, 0.4)
    silence_mean, _ = _mean_max_db(silent, 1.3, 0.4)
    assert loop_mean > -45
    assert silence_mean < -60


@pytest.mark.asyncio
async def test_video_bitstream_duration_and_frame_size_stay_put(tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", seconds=1.2, with_audio=True, size="160x120")
    music = _wav(tmp_path / "music.wav", seconds=3.0, rate=22050)
    before = video.read_bytes()
    video_probe = await probe_file(video)
    music_probe = await probe_file(music)
    result = await overlay_music(
        video_bytes=before,
        music_bytes=music.read_bytes(),
        video_probe=video_probe,
        music_probe=music_probe,
        settings=OverlaySettings(
            music_volume=0.3,
            original_audio_enabled=True,
            fade_in_seconds=0.1,
            fade_out_seconds=0.2,
        ),
        music_suffix=".wav",
    )
    assert before == video.read_bytes()
    out = tmp_path / "out.mp4"
    out.write_bytes(result.video_bytes)
    assert _video_md5(video) == _video_md5(out)
    probed = await probe_bytes(result.video_bytes, suffix=".mp4")
    assert probed.width == 160
    assert probed.height == 120
    assert probed.fps == pytest.approx(30, abs=0.1)
    assert probed.video_codec == "h264"
    assert probed.has_audio
    assert probed.duration_seconds == pytest.approx(video_probe.duration_seconds, abs=0.2)
    assert any(item.startswith("music_resampled:22050->") for item in result.warnings)


@pytest.mark.asyncio
async def test_original_audio_switch_and_quiet_music_is_not_boosted(tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", seconds=1.0, with_audio=True)
    music = _full_scale(tmp_path / "music.wav", frequency=880)
    video_probe = await probe_file(video)
    music_probe = await probe_file(music)

    kept = await overlay_music(
        video_bytes=video.read_bytes(),
        music_bytes=music.read_bytes(),
        video_probe=video_probe,
        music_probe=music_probe,
        settings=OverlaySettings(music_volume=0, original_audio_enabled=True),
        music_suffix=".wav",
    )
    dropped = await overlay_music(
        video_bytes=video.read_bytes(),
        music_bytes=music.read_bytes(),
        video_probe=video_probe,
        music_probe=music_probe,
        settings=OverlaySettings(music_volume=0, original_audio_enabled=False),
        music_suffix=".wav",
    )
    kept_path = tmp_path / "kept.mp4"
    dropped_path = tmp_path / "dropped.mp4"
    kept_path.write_bytes(kept.video_bytes)
    dropped_path.write_bytes(dropped.video_bytes)
    kept_mean, _ = _mean_max_db(kept_path, 0.1, 0.6)
    dropped_mean, _ = _mean_max_db(dropped_path, 0.1, 0.6)
    assert kept_mean > -45
    assert dropped_mean < -60

    quiet = await overlay_music(
        video_bytes=_video(tmp_path / "silent.mp4", seconds=1.0, with_audio=False).read_bytes(),
        music_bytes=music.read_bytes(),
        video_probe=await probe_file(tmp_path / "silent.mp4"),
        music_probe=music_probe,
        settings=OverlaySettings(
            music_volume=0.05,
            original_audio_enabled=False,
        ),
        music_suffix=".wav",
    )
    quiet_path = tmp_path / "quiet.mp4"
    quiet_path.write_bytes(quiet.video_bytes)
    _mean, quiet_max = _mean_max_db(quiet_path, 0.1, 0.6)
    assert quiet_max < -18


@pytest.mark.asyncio
async def test_mp3_and_m4a_music_decode(tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", seconds=0.8, with_audio=False)
    video_probe = await probe_file(video)
    rendered_any = False
    for suffix, codec in ((".mp3", "libmp3lame"), (".m4a", "aac")):
        music = tmp_path / f"song{suffix}"
        result_encode = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=660:duration=1.5:sample_rate=44100",
                "-c:a",
                codec,
                str(music),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result_encode.returncode != 0 or not music.exists():
            continue
        music_probe = await probe_file(music)
        rendered = await overlay_music(
            video_bytes=video.read_bytes(),
            music_bytes=music.read_bytes(),
            video_probe=video_probe,
            music_probe=music_probe,
            settings=OverlaySettings(
                music_volume=0.5,
                original_audio_enabled=False,
            ),
            music_suffix=suffix,
        )
        out = await probe_bytes(rendered.video_bytes, suffix=".mp4")
        assert out.has_audio
        assert out.width == video_probe.width
        rendered_any = True
    if not rendered_any:
        pytest.skip("neither libmp3lame nor aac encoder is available")
