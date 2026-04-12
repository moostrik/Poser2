---
description: "Use when editing pose Frame ECS data structures or pose feature definitions."
applyTo: "modules/pose/frame/**, modules/pose/features/**"
---
# Pose Data Guidelines

## Core contract

- These rules apply to pose `Frame` objects in the Frame ECS, not video/image frames
- `frame[FeatureType]` must never fail in normal flow
- Missing feature data is represented as NaN values with score `0.0`
- Consumers read directly and handle NaN, not presence checks

## Frames are immutable carriers

- `Frame` carries pose identity, timestamps, and typed feature data
- Do not mutate frame payloads in place; produce enriched output frames instead
- Preserve existing pose data unless a transformation intentionally replaces it

## Immutability and ownership

- Feature data is immutable after construction
- For pose features, constructors may take ownership of numpy arrays and mark them read-only
- Do not add defensive copies in hot paths unless correctness requires it
- Mutations should be functional (create a new frame / feature output)

## Features

- Add a new feature type only when it represents stable pose-domain data that should flow through the Frame ECS
- Choose `BaseScalarFeature` for one value per element and `BaseVectorFeature` for fixed-size vectors per element
- Implement enum mapping, range metadata, and dummy creation consistently
- Keep NaN and score semantics aligned: invalid or missing values must carry score `0.0`
- Export new features from the package so the rest of the pose pipeline can use them explicitly

## Performance

- Avoid Python loops over numeric arrays in hot code
- Prefer vectorized numpy operations
- Pre-allocate where it meaningfully reduces latency spikes

## Feature authoring

When adding a new feature:
1. Subclass the correct base (`BaseScalarFeature` or `BaseVectorFeature`)
2. Implement enum mapping, range metadata, and dummy creation
3. Export from module `__init__.py`
4. Keep ownership, immutability, and NaN semantics consistent with existing features
