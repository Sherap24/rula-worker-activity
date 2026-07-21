# CML ↔ MediaPipe Pose joint mapping reference

This document links **CML 3D skeleton joints** (15- and 20-node layouts) to **MediaPipe Pose** landmarks used in the Week 3 CWPV video pipeline.

**Scope:** Reference only. CML JSON is **not** processed with MediaPipe. CWPV videos use MediaPipe 2D/3D normalized landmarks.

## MediaPipe Pose landmarks (33 points)

MediaPipe uses a fixed 33-landmark body model. See [MediaPipe Pose landmark list](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker).

Key landmarks for construction activity and ergonomic screening:

| MediaPipe landmark | Use in this project |
|--------------------|---------------------|
| `NOSE`, `LEFT_EAR`, `RIGHT_EAR` | Head orientation proxy |
| `LEFT_SHOULDER`, `RIGHT_SHOULDER` | Trunk / shoulder line |
| `LEFT_ELBOW`, `RIGHT_ELBOW` | Arm flexion |
| `LEFT_WRIST`, `RIGHT_WRIST` | Hand height, reach |
| `LEFT_HIP`, `RIGHT_HIP` | Pelvis / body center |
| `LEFT_KNEE`, `RIGHT_KNEE` | Lower-limb flexion |
| `LEFT_ANKLE`, `RIGHT_ANKLE` | Foot placement |
| `LEFT_HEEL`, `RIGHT_HEEL`, `LEFT_FOOT_INDEX`, `RIGHT_FOOT_INDEX` | Zone-event foot point (Week 6) |

Body-center for zone events (Week 6): midpoint of `LEFT_HIP` and `RIGHT_HIP`, or hip–shoulder centroid.

## CML 15-node layout (approximate correspondence)

Defined in `configs/cml_skeleton_layouts.yaml`. Approximate MediaPipe analogs:

| CML 15-node joint | MediaPipe analog |
|-------------------|------------------|
| Head / head top | `NOSE` (proxy) |
| Neck | Shoulder midpoint |
| Spine / chest | Midpoint shoulders–hips |
| Left/right shoulder | `LEFT_SHOULDER`, `RIGHT_SHOULDER` |
| Left/right elbow | `LEFT_ELBOW`, `RIGHT_ELBOW` |
| Left/right hand | `LEFT_WRIST`, `RIGHT_WRIST` |
| Left/right hip | `LEFT_HIP`, `RIGHT_HIP` |
| Left/right knee | `LEFT_KNEE`, `RIGHT_KNEE` |
| Left/right foot | `LEFT_ANKLE`, `RIGHT_ANKLE` |

CML 15-node uses **foot** joints; MediaPipe uses **ankle/heel/toe** — expect systematic angle differences (see `reports/cml_skeleton_feature_comparison.md`).

## CML 20-node layout

Adds explicit `Hipcentre`, `Spine`, and **ankle** joints. Closer to MediaPipe lower-body topology but still 3D mocap-scale, not normalized image coordinates.

| CML 20-node joint | MediaPipe analog |
|-------------------|------------------|
| Hipcentre | Midpoint `LEFT_HIP`, `RIGHT_HIP` |
| Spine | Torso proxy between shoulders and hips |
| Left/right ankle | `LEFT_ANKLE`, `RIGHT_ANKLE` |

## Domain transfer note

CML skeletons are **mocap-style 3D** joint sequences. CWPV + MediaPipe produce **image-normalized** landmarks from real construction video. Do not assume feature parity across domains without explicit harmonization.

## Ergonomic screening (Week 6)

Joint angles for screening indicators will be computed from **MediaPipe landmarks on CWPV/local video**, not from CML JSON. CML angles remain useful for skeleton-only activity baselines.
