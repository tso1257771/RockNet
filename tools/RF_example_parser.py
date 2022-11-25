import os
import numpy as np
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

## order 1. write tfrecord from obspy traces
def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    # BytesList won't unpack a string from an EagerTensor.
    value = value.numpy() 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

## order 2. read tfrecord files
def write_TFRecord_RF_fusion(trc_3C, spectrogram,
        EQpick, EQmask, RFmask, idx, outfile):
    '''
    1. Create feature dictionary to be ready for setting up
        tf.train.Example object
        tf.train.Example can only accept 1-d data
    2. Create example protocol using tf.train.Example
    3. Write TFRecord object
    '''
    feature = {
        'trc_data': _float_feature(
            value=trc_3C.flatten()
        ),
        'spectrogram':  _float_feature(
            value=spectrogram.flatten()
        ),
        'EQpick': _float_feature(
            value=EQpick.flatten()
        ),
        'EQmask': _float_feature(
            value=EQmask.flatten()
        ),
        'RFmask': _float_feature(
            value=RFmask.flatten()
        ),
        'idx':_bytes_feature(
            value=idx.encode('utf-8')
        )
    }

    example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature)) 
    out = tf.io.TFRecordWriter(outfile)
    out.write(example_proto.SerializeToString())

def _parse_function_RF_fusion(
    record, data_length=6000, 
    spec_signal_shape=(51, 601, 2), 
    batch_size=10
):
    flatten_size_trc = data_length
    spec_size = spec_signal_shape[0]*\
        spec_signal_shape[1]*spec_signal_shape[2]

    feature = {
        "trc_data": tf.io.FixedLenFeature(
            [flatten_size_trc*3], tf.float32
        ),
        "spectrogram": tf.io.FixedLenFeature(
            [spec_size], tf.float32
        ),
        "EQpick": tf.io.FixedLenFeature(
            [flatten_size_trc*3], tf.float32
        ),
        "EQmask": tf.io.FixedLenFeature(
            [flatten_size_trc*2], tf.float32
        ),
        "RFmask": tf.io.FixedLenFeature(
            [flatten_size_trc*2], tf.float32
        ),
        "idx": tf.io.FixedLenFeature(
            [], tf.string
        )
    }

    record = tf.io.parse_example(record, feature)
    record['trc_data'] = tf.reshape(
        record['trc_data'], 
        (batch_size, data_length, 3))
    record['spectrogram'] = tf.reshape(
        record['spectrogram'], 
        (batch_size, 
        spec_signal_shape[0],
        spec_signal_shape[1],
        spec_signal_shape[2])
    )
    record['EQpick'] = tf.reshape(
        record['EQpick'],
        (batch_size, data_length, 3)
    )
    record['EQmask'] = tf.reshape(
        record['EQmask'],
        (batch_size, data_length, 2)
    )                    
    record['RFmask'] = tf.reshape(
        record['RFmask'],
        (batch_size, data_length, 2)
    )  

    return record['trc_data'], record['spectrogram'],\
        record['EQpick'], record['EQmask'], \
        record['RFmask'], record['idx']

def tfrecord_dataset_fusion_RF(
    file_list, repeat=-1, 
    batch_size=None, 
    data_length=6000, 
    spec_signal_shape=(51, 601, 2),
    shuffle_buffer_size=300
):
    if batch_size == None:
        raise ValueError("Must specify value of `batch_size`")
    else:
        dataset = tf.data.TFRecordDataset(
            file_list, 
            num_parallel_reads=AUTOTUNE
        )
        if shuffle_buffer_size:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size, 
                reshuffle_each_iteration=True
            )
        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        parsed_dataset = dataset.map(
            lambda x:_parse_function_RF_fusion(
                x,
                data_length=data_length,
                spec_signal_shape=spec_signal_shape, 
                batch_size=batch_size
            ), 
            num_parallel_calls=AUTOTUNE
        )
        return parsed_dataset