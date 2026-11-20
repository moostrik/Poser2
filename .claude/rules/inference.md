---
description: "The inference package: GPU streams, callbacks, model files"
paths:
  - "modules/inference/**"
---
# Inference

- A reader inside the package lets its own stream wait for the item's event; the thread itself does not wait
- A tensor read on another stream than the one that made it is marked there (`record_stream`)
- A producer with readers inside and outside the package calls the inside ones first, then waits for its stream and calls its finished callbacks
- A model file's name says `_fp16` when it is FP16
