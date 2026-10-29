---
description: Pose nodes, trackers, windows, analytics and the pipeline builder
paths:
  - "modules/pose/trackers/**"
  - "modules/pose/window/**"
  - "modules/pose/analytics/**"
  - "modules/pose/pipeline/**"
  - "apps/*/pose/pipeline.py"
---
# Pose Processing Guidelines

## Pipeline

- An app declares its stages in `apps/<app>/pose/pipeline.py` as a `PipelineSpec`; `main.py` builds the `PosePipeline`, connects the clocks and subscribes the board, the recorders and OSC
- Only sources create poses; a stage's steps never add or remove one
- A stage's interpolators are the one crossing from the frame domain to an output domain; a tick source joins at a stage the tick drives
- A step entry is plain data: a node class with its settings object, or a node built once; no closures
- A node takes time from the frame it processes; the crossing stamps a frame with the tick it was made on, so no node reads a clock
- A stage emits a group record (`GroupRecord`) beside its poses: what belongs to everyone in the frame and to no single pose (a level, a gate, this tick's hits); read it with `add_group_callback` or the board's `get_group`, on the same tick as the poses
- The group record follows the poses through the stages of a domain; across the crossing and an async step the latest record is held
- A value derived from pose data, a threshold on it included (a gate, a dead zone), is a pipeline step with its settings; the show reads it from the record or the poses

## Nodes

- Extractors derive new features from existing pose features and return enriched frames
- Filters preserve frame identity while refining feature values over time
- Group nodes see every pose of a frame: `process` stamps the poses with what they derive from the group, `record` writes the group record; they never add or remove a pose
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
