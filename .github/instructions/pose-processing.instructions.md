---
description: "Use when editing pose nodes, trackers, windows, or batch processing."
applyTo: "modules/pose/nodes/**, modules/pose/trackers/**, modules/pose/window/**, modules/pose/batch/**"
---
# Pose Processing Guidelines

## Nodes

- Extractors derive new features from existing pose features and return enriched frames
- Filters preserve frame identity while refining feature values over time
- Applicators should compute explicit pose-domain outputs rather than introducing hidden side channels
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
