from src.features.subject_selection import select_dominant, subject_score
from tests.conftest import make_box

CFG = {"w_area": 0.4, "w_center_prior": 0.3, "w_confidence": 0.3,
       "person_boost": 1.2, "person_terms": ["person", "face"], "max_area": 0.9}


def test_whole_frame_box_excluded():
    frame_filler = {"box": [0.0, 0.0, 1.0, 0.98], "score": 0.95,
                    "phrase": "building"}
    small = make_box(0.3, 0.4, 0.15, 0.3, 0.6, "person")
    sel = select_dominant([frame_filler, small], CFG)
    assert sel["subject"] is small
    sel_only_filler = select_dominant([frame_filler], CFG)
    assert sel_only_filler["no_subject"]


def test_empty_boxes_no_subject():
    sel = select_dominant([], CFG)
    assert sel["no_subject"] and sel["subject"] is None and sel["face"] is None


def test_person_boost_beats_equal_object():
    person = make_box(0.5, 0.5, 0.2, 0.2, 0.8, "person")
    obj = make_box(0.5, 0.5, 0.2, 0.2, 0.8, "vehicle")
    assert subject_score(person, CFG) > subject_score(obj, CFG)
    sel = select_dominant([obj, person], CFG)
    assert sel["subject"]["phrase"] == "person"


def test_center_prior_prefers_central_box():
    central = make_box(0.5, 0.5, 0.2, 0.2, 0.7, "vehicle")
    corner = make_box(0.1, 0.1, 0.2, 0.2, 0.7, "vehicle")
    sel = select_dominant([corner, central], CFG)
    assert sel["subject"] is central


def test_face_captured_separately():
    body = make_box(0.5, 0.55, 0.3, 0.6, 0.9, "person")
    face = make_box(0.5, 0.3, 0.1, 0.12, 0.85, "face")
    sel = select_dominant([body, face], CFG)
    assert sel["face"] is face
    assert sel["subject"] is body  # larger area wins overall
