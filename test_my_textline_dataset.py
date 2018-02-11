import tensorflow as tf

from my_textline_dataset import MyTextLineDataset


class CustomDatasetTest(tf.test.TestCase):
  def testMyTextLineDataset(self):
    dataset = MyTextLineDataset('lines.txt')
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as session:
      print(session.run(next_element))


if __name__ == '__main__':
  tf.test.main()
