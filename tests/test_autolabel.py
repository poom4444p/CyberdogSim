"""Tests for sim/autolabel.py: walked path -> VAMOS training pair.

No scene and no torch: the labeller is geometry plus a file format, and both
can be checked against the projector it shares with the live loop.
"""
import json
import math

import numpy as np
import pytest

from cyberdog.planning.checkpoint_projector import (load_camera_config,
                                                    project_to_pixel, vamos_prompt)
from cyberdog.sim import autolabel as al

CAM = load_camera_config()


def straight(length=4.0, step=0.02):
    return [(k * step, 0.0) for k in range(int(length / step) + 1)]


def test_along_is_even_by_arc_length_and_skips_standing_still():
    track = [(0.0, 0.0)] * 30 + [(x, 0.0) for x in np.linspace(0, 1, 11)]
    pts = al.along(track, 1.0, step=0.25)
    assert [round(p[0], 6) for p in pts] == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_along_stops_at_length():
    pts = al.along(straight(4.0), 2.0, step=0.5)
    assert math.isclose(pts[-1][0], 2.0, abs_tol=1e-9)


def test_straight_ahead_label_is_on_the_centre_column_and_climbs_the_image():
    # Lens at the origin facing +x: a path straight down the corridor.
    got = al.label(straight(), (0.0, 0.0, 0.0), CAM, 3.0)
    assert got is not None
    pixels, locs, ground = got
    assert len(pixels) == len(locs) == len(ground) == al.N_POINTS
    assert all(u == int(CAM["cx"]) for u, _ in pixels)
    vs = [v for _, v in pixels]
    assert vs == sorted(vs, reverse=True)          # further = higher in frame
    assert math.isclose(ground[-1][0], 3.0, abs_tol=al.STEP)


def test_label_matches_the_projector_the_live_loop_uses():
    track = [(t, 0.3 * math.sin(t)) for t in np.linspace(0, 4, 200)]
    pose = (-0.2, 0.05, 0.1)
    pixels, locs, ground = al.label(track, pose, CAM, 3.5)
    for g, px, lc in zip(ground, pixels, locs):
        r = project_to_pixel(g, pose, CAM)
        assert r["pixel"] == px and r["loc"] == lc


def test_path_is_cut_where_it_leaves_the_frame():
    # 1.5 m ahead, then a hard left out of view: only the visible part counts.
    track = straight(1.5) + [(1.5, 0.02 * k) for k in range(1, 150)]
    _, _, ground = al.label(track, (0.0, 0.0, 0.0), CAM, 4.0)
    assert max(x for x, _ in ground) <= 1.5 + 1e-6
    assert all(project_to_pixel(g, (0.0, 0.0, 0.0), CAM)["state"] == "TRACK"
               for g in ground)


def test_too_little_in_view_is_no_label():
    # Turning on the spot: nothing walked, nothing to draw.
    assert al.label([(0.0, 0.0)] * 50, (0.0, 0.0, 0.0), CAM, 3.0) is None
    # Walking straight out of the side of the frame.
    side = [(0.0, 0.02 * k) for k in range(200)]
    assert al.label(side, (0.0, 0.0, 0.0), CAM, 3.0) is None


def test_prompt_is_the_one_the_live_loop_sends():
    assert al.prompt((444, 526)) == vamos_prompt({"loc": (444, 526)})


def test_target_decodes_the_way_the_server_reads_it():
    import re
    locs = [(512, 900), (510, 800), (505, 700), (500, 650), (498, 620)]
    t = al.target(locs)
    nums = [int(x[4:8]) for x in re.findall(r"<loc\d{4}>", t)]
    assert list(zip(nums[0::2], nums[1::2])) == locs


def test_recorder_writes_labelled_frames_and_drops_the_rest(tmp_path):
    rec = al.AutoLabel(str(tmp_path), CAM)
    rec.start_leg(2)
    img = np.zeros((CAM["height"], CAM["width"], 3), np.uint8)
    track = straight(6.0, step=0.04)
    for n, (x, y) in enumerate(track):
        state = {"state": "TRACK", "destination": (x + 3.0, 0.0), "loc": (512, 600)}
        if n > len(track) - 40:
            state = {"state": "ALIGN", "turn": "left"}
        rec.tick(n, (x, y, 0.0), (x + 0.1, y, 0.0), state, lambda: img)
    kept = rec.finish({"outcome": "ARRIVED", "clean": True})
    lines = [json.loads(l) for l in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert kept == len(lines) > 0
    frames = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert frames == sorted(l["frame"].split("/")[1] for l in lines)
    # Near the end of the leg less than MIN_LENGTH is left to walk: no label,
    # and its frame is gone too.
    assert all(l["tick"] < len(track) - 25 for l in lines)
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["clean"] and run["samples"] == kept
    for l in lines:
        assert l["prompt"].startswith("Navigate to x=<loc")
        assert l["target"].count("<loc") == 2 * al.N_POINTS


def test_standing_still_keeps_one_frame_not_many(tmp_path):
    rec = al.AutoLabel(str(tmp_path), CAM)
    rec.start_leg(1)
    img = np.zeros((CAM["height"], CAM["width"], 3), np.uint8)
    calls = []
    def image():
        calls.append(1)
        return img
    state = {"state": "TRACK", "destination": (3.0, 0.0), "loc": (512, 600)}
    for n in range(200):
        rec.tick(n, (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), state, image)
    assert len(calls) == 1
