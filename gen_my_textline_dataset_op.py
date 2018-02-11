import tensorflow as tf

from tensorflow.python.eager import context as _context
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes

from tensorflow.python.framework import ops as _ops
from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
from tensorflow.core.framework import types_pb2 as _types_pb2
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import op_def_library as _op_def_library

import os


lib = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      "my_line_dataset.so"))

#  copy-pasta'd from gen_dataset_ops.py within tensorflow (note: generated file)
def my_text_line_dataset(filenames, compression_type, buffer_size, name=None):
  r"""Creates a dataset that emits the lines of one or more text files.

  Args:
    filenames: A `Tensor` of type `string`.
      A scalar or a vector containing the name(s) of the file(s) to be
      read.
    compression_type: A `Tensor` of type `string`.
      A scalar containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    buffer_size: A `Tensor` of type `int64`.
      A scalar containing the number of bytes to buffer.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "MyTextLineDataset", filenames=filenames,
        compression_type=compression_type, buffer_size=buffer_size, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
    compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
    buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
    _inputs_flat = [filenames, compression_type, buffer_size]
    _attrs = None
    _result = _execute.execute(b"MyTextLineDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "MyTextLineDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def _create_my_line_dataset():
  op = _op_def_pb2.OpDef()
  op.name = "MyTextLineDataset"

  filenames = _op_def_pb2.OpDef.ArgDef()
  filenames.name = "filenames"
  filenames.type = _types_pb2.DT_STRING
  compression_type = _op_def_pb2.OpDef.ArgDef()
  compression_type.name = "compression_type"
  compression_type.type = _types_pb2.DT_STRING
  buffer_size = _op_def_pb2.OpDef.ArgDef()
  buffer_size.name = "buffer_size"
  buffer_size.type = _types_pb2.DT_INT64
  op.input_arg.extend([filenames, compression_type, buffer_size])

  handle = _op_def_pb2.OpDef.ArgDef()
  handle.name = "handle"
  handle.type = _types_pb2.DT_VARIANT
  op.output_arg.extend([handle])

  op.is_stateful = True

  return op


def _create_op_def_library(op_protos):
  op_list = _op_def_pb2.OpList()

  for op_proto in op_protos:
    registered_ops = _op_def_registry.get_registered_ops()
    if op_proto.name not in registered_ops:
      raise LookupError("Op with name {0} not registered".format(op_proto.name))

    op_list.op.extend([op_proto])

  # Fails if the interfaces ("op schemas") don't match between the
  # previously registered op and this one.
  _op_def_registry.register_op_list(op_list)

  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_op_def_lib = _create_op_def_library([_create_my_line_dataset()])
