	�<+iŏ @�<+iŏ @!�<+iŏ @	jsݑh��?jsݑh��?!jsݑh��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�<+iŏ @�uX��?A��$W�@Y8�a�A
�?*	�V1c@2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[1]::ConcatenateSB��^~�?!!�{O'D@)�;���?1�\�McB@:Preprocessing2F
Iterator::ModelY�;ۣ7�?!&�����A@)�E|'f��?1��'l�K5@:Preprocessing2S
Iterator::Model::ParallelMapkg{��?!����3-@)kg{��?1����3-@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatx���Ĭ�?!`���.@)%�s}�?1�r���&@:Preprocessing2X
!Iterator::Model::ParallelMap::ZipyW=`2�?!m.�!�P@)����#*�?1���Lݦ@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensorH�Sȕzv?!���ʆ�@)H�Sȕzv?1���ʆ�@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�\����?!"����PE@)<-?p�'p?1�$��@:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorGsd��h?!V���.~�?)Gsd��h?1V���.~�?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice@��wԘ`?!1�K:�?)@��wԘ`?11�K:�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�uX��?�uX��?!�uX��?      ��!       "      ��!       *      ��!       2	��$W�@��$W�@!��$W�@:      ��!       B      ��!       J	8�a�A
�?8�a�A
�?!8�a�A
�?R      ��!       Z	8�a�A
�?8�a�A
�?!8�a�A
�?JCPU_ONLY