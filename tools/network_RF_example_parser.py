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
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_TFRecord_RF_network(
        net_data, RF_detect, EQ_detect, outfile):
    '''
    1. Create feature dictionary to be ready for setting up
        tf.train.Example object
        tf.train.Example can only accept 1-d data
    2. Create example protocol using tf.train.Example
    3. Write TFRecord object
    '''
    [sta1_trc_3C, sta1_spectrogram,
        sta1_EQpick, sta1_EQmask, sta1_RFmask, 
        sta2_trc_3C, sta2_spectrogram,
        sta2_EQpick, sta2_EQmask, sta2_RFmask,         
        sta3_trc_3C, sta3_spectrogram,
        sta3_EQpick, sta3_EQmask, sta3_RFmask,         
        sta4_trc_3C, sta4_spectrogram,
        sta4_EQpick, sta4_EQmask, sta4_RFmask
    ] = net_data

    feature = {
        'sta1_trc_data': _float_feature(value=sta1_trc_3C.flatten() ),
        'sta1_spectrogram':  _float_feature(value=sta1_spectrogram.flatten() ),
        'sta1_EQpick': _float_feature(value=sta1_EQpick.flatten()),
        'sta1_EQmask': _float_feature(value=sta1_EQmask.flatten()),
        'sta1_RFmask': _float_feature(value=sta1_RFmask.flatten()),

        'sta2_trc_data': _float_feature(value=sta2_trc_3C.flatten() ),
        'sta2_spectrogram':  _float_feature(value=sta2_spectrogram.flatten() ),
        'sta2_EQpick': _float_feature(value=sta2_EQpick.flatten()),
        'sta2_EQmask': _float_feature(value=sta2_EQmask.flatten()),
        'sta2_RFmask': _float_feature(value=sta2_RFmask.flatten()),

        'sta3_trc_data': _float_feature(value=sta3_trc_3C.flatten() ),
        'sta3_spectrogram':  _float_feature(value=sta3_spectrogram.flatten() ),
        'sta3_EQpick': _float_feature(value=sta3_EQpick.flatten()),
        'sta3_EQmask': _float_feature(value=sta3_EQmask.flatten()),
        'sta3_RFmask': _float_feature(value=sta3_RFmask.flatten()),     

        'sta4_trc_data': _float_feature(value=sta4_trc_3C.flatten() ),
        'sta4_spectrogram':  _float_feature(value=sta4_spectrogram.flatten() ),
        'sta4_EQpick': _float_feature(value=sta4_EQpick.flatten()),
        'sta4_EQmask': _float_feature(value=sta4_EQmask.flatten()),
        'sta4_RFmask': _float_feature(value=sta4_RFmask.flatten()),

        'RF_detect': _float_feature(value=RF_detect.flatten()),
        'EQ_detect': _float_feature(value=EQ_detect.flatten())
        }

    example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature)) 
    out = tf.io.TFRecordWriter(outfile)
    out.write(example_proto.SerializeToString())

def _parse_function_RF_network(record, data_length=6000, 
        spec_signal_shape=(51, 601, 2), batch_size=10):
    flatten_size_trc = data_length
    spec_size = spec_signal_shape[0]*spec_signal_shape[1]*spec_signal_shape[2]

    feature = {
        "sta1_trc_data": tf.io.FixedLenFeature([flatten_size_trc*3], tf.float32),
        "sta1_spectrogram": tf.io.FixedLenFeature([spec_size], tf.float32),
        "sta1_EQpick": tf.io.FixedLenFeature([flatten_size_trc*3], tf.float32),
        "sta1_EQmask": tf.io.FixedLenFeature([flatten_size_trc*2], tf.float32),
        "sta1_RFmask": tf.io.FixedLenFeature([flatten_size_trc*2], tf.float32),

        "sta2_trc_data": tf.io.FixedLenFeature([flatten_size_trc*3], tf.float32),
        "sta2_spectrogram": tf.io.FixedLenFeature([spec_size], tf.float32),
        "sta2_EQpick": tf.io.FixedLenFeature([flatten_size_trc*3], tf.float32),
        "sta2_EQmask": tf.io.FixedLenFeature([flatten_size_trc*2], tf.float32),
        "sta2_RFmask": tf.io.FixedLenFeature([flatten_size_trc*2], tf.float32),

        "sta3_trc_data": tf.io.FixedLenFeature([flatten_size_trc*3], tf.float32),
        "sta3_spectrogram": tf.io.FixedLenFeature([spec_size], tf.float32),
        "sta3_EQpick": tf.io.FixedLenFeature([flatten_size_trc*3], tf.float32),
        "sta3_EQmask": tf.io.FixedLenFeature([flatten_size_trc*2], tf.float32),
        "sta3_RFmask": tf.io.FixedLenFeature([flatten_size_trc*2], tf.float32),

        "sta4_trc_data": tf.io.FixedLenFeature([flatten_size_trc*3], tf.float32),
        "sta4_spectrogram": tf.io.FixedLenFeature([spec_size], tf.float32),
        "sta4_EQpick": tf.io.FixedLenFeature([flatten_size_trc*3], tf.float32),
        "sta4_EQmask": tf.io.FixedLenFeature([flatten_size_trc*2], tf.float32),
        "sta4_RFmask": tf.io.FixedLenFeature([flatten_size_trc*2], tf.float32),

        "RF_detect": tf.io.FixedLenFeature([flatten_size_trc*2], tf.float32),
        "EQ_detect": tf.io.FixedLenFeature([flatten_size_trc*2], tf.float32)
    }

    record = tf.io.parse_example(record, feature)
    # station 1
    record['sta1_trc_data'] = tf.reshape(record['sta1_trc_data'], 
                    (batch_size, data_length, 3))
    record['sta1_spectrogram'] = tf.reshape(record['sta1_spectrogram'], 
                    (batch_size, 
                    spec_signal_shape[0],
                    spec_signal_shape[1],
                    spec_signal_shape[2])
                )
    record['sta1_EQpick'] = tf.reshape(record['sta1_EQpick'],
                    (batch_size, data_length, 3))
    record['sta1_EQmask'] = tf.reshape(record['sta1_EQmask'],
                    (batch_size, data_length, 2))                    
    record['sta1_RFmask'] = tf.reshape(record['sta1_RFmask'],
                    (batch_size, data_length, 2))  
    # station 2
    record['sta2_trc_data'] = tf.reshape(record['sta2_trc_data'], 
                    (batch_size, data_length, 3))
    record['sta2_spectrogram'] = tf.reshape(record['sta2_spectrogram'], 
                    (batch_size, 
                    spec_signal_shape[0],
                    spec_signal_shape[1],
                    spec_signal_shape[2])
                )
    record['sta2_EQpick'] = tf.reshape(record['sta2_EQpick'],
                    (batch_size, data_length, 3))
    record['sta2_EQmask'] = tf.reshape(record['sta2_EQmask'],
                    (batch_size, data_length, 2))                    
    record['sta2_RFmask'] = tf.reshape(record['sta2_RFmask'],
                    (batch_size, data_length, 2))  
    # station 3
    record['sta3_trc_data'] = tf.reshape(record['sta3_trc_data'], 
                    (batch_size, data_length, 3))
    record['sta3_spectrogram'] = tf.reshape(record['sta3_spectrogram'], 
                    (batch_size, 
                    spec_signal_shape[0],
                    spec_signal_shape[1],
                    spec_signal_shape[2])
                )
    record['sta3_EQpick'] = tf.reshape(record['sta3_EQpick'],
                    (batch_size, data_length, 3))
    record['sta3_EQmask'] = tf.reshape(record['sta3_EQmask'],
                    (batch_size, data_length, 2))                    
    record['sta3_RFmask'] = tf.reshape(record['sta3_RFmask'],
                    (batch_size, data_length, 2))  
    # station 4
    record['sta4_trc_data'] = tf.reshape(record['sta4_trc_data'], 
                    (batch_size, data_length, 3))
    record['sta4_spectrogram'] = tf.reshape(record['sta4_spectrogram'], 
                    (batch_size, 
                    spec_signal_shape[0],
                    spec_signal_shape[1],
                    spec_signal_shape[2])
                )
    record['sta4_EQpick'] = tf.reshape(record['sta4_EQpick'],
                    (batch_size, data_length, 3))
    record['sta4_EQmask'] = tf.reshape(record['sta4_EQmask'],
                    (batch_size, data_length, 2))                    
    record['sta4_RFmask'] = tf.reshape(record['sta4_RFmask'],
                    (batch_size, data_length, 2))

    record['RF_detect'] = tf.reshape(record['RF_detect'],
                    (batch_size, data_length, 2))

    record['EQ_detect'] = tf.reshape(record['EQ_detect'],
                    (batch_size, data_length, 2))
    # change the axis from 
    # [station_number, batch_size, data_length, data_channel] to
    # [batch_size, station_number, data_length, data_channel] to
    trc_tensor = tf.transpose(
        tf.stack([
            record['sta1_trc_data'], record['sta2_trc_data'],
            record['sta3_trc_data'], record['sta4_trc_data']
        ]), [1, 0, 2, 3]
    )
    spec_tensor = tf.transpose(
        tf.stack([
        record['sta1_spectrogram'], record['sta2_spectrogram'],
        record['sta3_spectrogram'], record['sta4_spectrogram']
        ]), 
        [1, 0, 2, 3, 4]
    )
    eqpick_tensor = tf.transpose(
        tf.stack([
        record['sta1_EQpick'], record['sta2_EQpick'],
        record['sta3_EQpick'], record['sta4_EQpick']
        ]),  [1, 0, 2, 3]
    )
    eqmask_tensor = tf.transpose(
        tf.stack([
        record['sta1_EQmask'], record['sta2_EQmask'],
        record['sta3_EQmask'], record['sta4_EQmask']
        ]),  [1, 0, 2, 3]
    )
    rfmask_tensor = tf.transpose(
        tf.stack([
        record['sta1_RFmask'], record['sta2_RFmask'],
        record['sta3_RFmask'], record['sta4_RFmask']
        ]),  [1, 0, 2, 3]
    )

    rf_occ_tensor = record["RF_detect"]
    eq_occ_tensor = record["EQ_detect"]

    return trc_tensor, spec_tensor, eqpick_tensor, eqmask_tensor,\
         rfmask_tensor, eq_occ_tensor, rf_occ_tensor

def tfrecord_dataset_fusion_RF_net(file_list, repeat=-1, batch_size=None, 
                        data_length=6000, spec_signal_shape=(51, 601, 2),
                        shuffle_buffer_size=300, shuffle=True):
    if batch_size == None:
        raise ValueError("Must specify value of `batch_size`")
    else:
        dataset = tf.data.TFRecordDataset(file_list, 
                    num_parallel_reads=AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, 
                        reshuffle_each_iteration=True)
        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        parsed_dataset = dataset.map(
                lambda x:_parse_function_RF_network(x,
                 data_length=data_length,
                 spec_signal_shape=spec_signal_shape, 
                 batch_size=batch_size),
                 num_parallel_calls=AUTOTUNE)
        return parsed_dataset