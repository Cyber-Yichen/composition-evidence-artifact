from __future__ import annotations


def headroom(face_box: list[float] | None) -> dict:
    if face_box is None:
        return {"face_top_y": None, "headroom": None}
    top = face_box[1]
    return {"face_top_y": top, "headroom": top}
