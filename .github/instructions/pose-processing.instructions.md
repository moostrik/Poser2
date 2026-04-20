---
description: "Use when editing pose nodes, trackers, windows, or batch processing."
applyTo: "modules/pose/nodes/**, modules/pose/trackers/**, modules/pose/window/**, modules/pose/batch/**"
---
# Pose Processing Guidelines

## Processing pipeline contract

- Pose processing operates on pose `Frame` objects and must preserve the pose data contract
- Prefer explicit pipeline stages over hidden side effects between components
- Enrich or transform frames functionally rather than mutating shared state

## Nodes

- Extractors derive new features from existing pose features and return enriched frames
- Filters preserve frame identity while refining feature values over time
- Applicators should compute explicit pose-domain outputs rather than introducing hidden side channels
- Interpolators and window nodes own internal temporal state and must reset cleanly

## Trackers

- Trackers maintain independent per-`track_id` state
- Reset tracker state when a pose is lost or no longer present
- One track failing should not break processing for other tracks
- Keep callback fan-out explicit and isolated from tracker state transitions

## Windowing

- Feature windows must stay consistent with frame and feature contracts
- Be explicit about partial versus full window emission behavior
- Reset temporal buffers when track continuity is broken

## Batch processing

- Batch code should serve pose pipeline throughput without hiding data ownership rules
- Keep queueing and buffering decisions aligned with low-latency priorities
- Prefer predictable data flow over opaque background processing
