from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from gen_my_textline_dataset_op import my_text_line_dataset


_DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024  # 256 KB


class MyTextLineDataset(Dataset):
  """A `Dataset` comprising lines from one or more text files."""

  def __init__(self, filenames, compression_type=None, buffer_size=None):
    """Creates a `TextLineDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer. A value of 0 results in the default buffering values chosen
        based on the compression type.
    """
    super(MyTextLineDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size", buffer_size, _DEFAULT_READER_BUFFER_SIZE_BYTES)

  def _as_variant_tensor(self):
    return my_text_line_dataset(self._filenames, self._compression_type,
                                self._buffer_size)

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.scalar()

  @property
  def output_types(self):
    return dtypes.string
