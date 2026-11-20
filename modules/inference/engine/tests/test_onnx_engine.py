"""OnnxEngine on a stub model (ONNX Runtime's CUDA provider): given `max_batch`, a smaller batch runs padded to it and
its outputs come back cut to the batch."""

import os
import tempfile
import unittest

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from ..onnx_engine import OnnxEngine


def write_stub_model(path: str) -> None:
    """x (N, 3) -> doubled (N, 3), x times two, and seen (N, 1), the batch N the model ran at in every row."""
    nodes = [
        helper.make_node('Mul', ['x', 'two'], ['doubled']),
        helper.make_node('Shape', ['x'], ['shape']),
        helper.make_node('Gather', ['shape', 'zero'], ['batch'], axis=0),
        helper.make_node('Cast', ['batch'], ['batch_float'], to=TensorProto.FLOAT),
        helper.make_node('Concat', ['batch', 'one'], ['seen_shape'], axis=0),
        helper.make_node('Expand', ['batch_float', 'seen_shape'], ['seen']),
    ]
    graph = helper.make_graph(
        nodes, 'stub_batch',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['N', 3])],
        [helper.make_tensor_value_info('doubled', TensorProto.FLOAT, ['N', 3]),
         helper.make_tensor_value_info('seen', TensorProto.FLOAT, ['N', 1])],
        initializer=[
            numpy_helper.from_array(np.array(2.0, dtype=np.float32), 'two'),
            numpy_helper.from_array(np.array([0], dtype=np.int64), 'zero'),
            numpy_helper.from_array(np.array([1], dtype=np.int64), 'one'),
        ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.save(model, path)


class PaddingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path: str = os.path.join(tempfile.mkdtemp(), 'stub_batch.onnx')
        write_stub_model(self.path)
        self.x: torch.Tensor = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3)

    def run_engine(self, max_batch: int | None) -> dict[str, torch.Tensor]:
        engine = OnnxEngine(self.path, max_batch)
        stream = torch.cuda.Stream()
        engine.setup(stream)
        outputs = engine.run({'x': self.x})
        stream.synchronize()
        return outputs

    def test_a_smaller_batch_runs_at_max_batch_and_comes_back_cut_to_its_size(self) -> None:
        outputs = self.run_engine(max_batch=4)
        self.assertEqual(tuple(outputs['doubled'].shape), (2, 3))
        self.assertTrue(torch.equal(outputs['doubled'], self.x * 2))
        self.assertEqual(outputs['seen'].flatten().tolist(), [4.0, 4.0])

    def test_without_max_batch_the_batch_runs_as_it_is(self) -> None:
        outputs = self.run_engine(max_batch=None)
        self.assertEqual(outputs['seen'].flatten().tolist(), [2.0, 2.0])


if __name__ == '__main__':
    unittest.main()
