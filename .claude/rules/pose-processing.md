---
description: Pose nodes, trackers, windows, and analytics
paths:
  - "modules/pose/trackers/**"
  - "modules/pose/window/**"
  - "modules/pose/analytics/**"
---
# Pose Processing Guidelines

## Nodes

- Extractors derive new features from existing pose features and return enriched frames
- Filters preserve frame identity while refining feature values over time
- Group nodes see every pose of a frame and stamp what they derive from the group; they never add or remove a pose
- Interpolators and window nodes own internal temporal state and must reset cleanly

## Trackers

- Trackers maintain independent per-`track_id` state
- Reset tracker state when a pose is lost or no longer present
- One track failing should not break processing for other tracks
- Keep callback fan-out isolated from tracker state transitions

## Windowing

- Be explicit about partial versus full window emission behavior
- Reset temporal buffers when track continuity is broken

## Batch processing

- Prefer predictable data flow over opaque background processing
