# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorboard/compat/proto/struct.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorboard.compat.proto import tensor_pb2 as tensorboard_dot_compat_dot_proto_dot_tensor__pb2
from tensorboard.compat.proto import tensor_shape_pb2 as tensorboard_dot_compat_dot_proto_dot_tensor__shape__pb2
from tensorboard.compat.proto import types_pb2 as tensorboard_dot_compat_dot_proto_dot_types__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n%tensorboard/compat/proto/struct.proto\x12\x0btensorboard\x1a%tensorboard/compat/proto/tensor.proto\x1a+tensorboard/compat/proto/tensor_shape.proto\x1a$tensorboard/compat/proto/types.proto\"\xfd\x05\n\x0fStructuredValue\x12,\n\nnone_value\x18\x01 \x01(\x0b\x32\x16.tensorboard.NoneValueH\x00\x12\x17\n\rfloat64_value\x18\x0b \x01(\x01H\x00\x12\x15\n\x0bint64_value\x18\x0c \x01(\x12H\x00\x12\x16\n\x0cstring_value\x18\r \x01(\tH\x00\x12\x14\n\nbool_value\x18\x0e \x01(\x08H\x00\x12;\n\x12tensor_shape_value\x18\x1f \x01(\x0b\x32\x1d.tensorboard.TensorShapeProtoH\x00\x12\x33\n\x12tensor_dtype_value\x18  \x01(\x0e\x32\x15.tensorboard.DataTypeH\x00\x12\x39\n\x11tensor_spec_value\x18! \x01(\x0b\x32\x1c.tensorboard.TensorSpecProtoH\x00\x12\x35\n\x0ftype_spec_value\x18\" \x01(\x0b\x32\x1a.tensorboard.TypeSpecProtoH\x00\x12H\n\x19\x62ounded_tensor_spec_value\x18# \x01(\x0b\x32#.tensorboard.BoundedTensorSpecProtoH\x00\x12,\n\nlist_value\x18\x33 \x01(\x0b\x32\x16.tensorboard.ListValueH\x00\x12.\n\x0btuple_value\x18\x34 \x01(\x0b\x32\x17.tensorboard.TupleValueH\x00\x12,\n\ndict_value\x18\x35 \x01(\x0b\x32\x16.tensorboard.DictValueH\x00\x12\x39\n\x11named_tuple_value\x18\x36 \x01(\x0b\x32\x1c.tensorboard.NamedTupleValueH\x00\x12\x30\n\x0ctensor_value\x18\x37 \x01(\x0b\x32\x18.tensorboard.TensorProtoH\x00\x12/\n\x0bnumpy_value\x18\x38 \x01(\x0b\x32\x18.tensorboard.TensorProtoH\x00\x42\x06\n\x04kind\"\x0b\n\tNoneValue\"9\n\tListValue\x12,\n\x06values\x18\x01 \x03(\x0b\x32\x1c.tensorboard.StructuredValue\":\n\nTupleValue\x12,\n\x06values\x18\x01 \x03(\x0b\x32\x1c.tensorboard.StructuredValue\"\x8c\x01\n\tDictValue\x12\x32\n\x06\x66ields\x18\x01 \x03(\x0b\x32\".tensorboard.DictValue.FieldsEntry\x1aK\n\x0b\x46ieldsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12+\n\x05value\x18\x02 \x01(\x0b\x32\x1c.tensorboard.StructuredValue:\x02\x38\x01\"E\n\tPairValue\x12\x0b\n\x03key\x18\x01 \x01(\t\x12+\n\x05value\x18\x02 \x01(\x0b\x32\x1c.tensorboard.StructuredValue\"G\n\x0fNamedTupleValue\x12\x0c\n\x04name\x18\x01 \x01(\t\x12&\n\x06values\x18\x02 \x03(\x0b\x32\x16.tensorboard.PairValue\"s\n\x0fTensorSpecProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12,\n\x05shape\x18\x02 \x01(\x0b\x32\x1d.tensorboard.TensorShapeProto\x12$\n\x05\x64type\x18\x03 \x01(\x0e\x32\x15.tensorboard.DataType\"\xd0\x01\n\x16\x42oundedTensorSpecProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12,\n\x05shape\x18\x02 \x01(\x0b\x32\x1d.tensorboard.TensorShapeProto\x12$\n\x05\x64type\x18\x03 \x01(\x0e\x32\x15.tensorboard.DataType\x12)\n\x07minimum\x18\x04 \x01(\x0b\x32\x18.tensorboard.TensorProto\x12)\n\x07maximum\x18\x05 \x01(\x0b\x32\x18.tensorboard.TensorProto\"\xfa\x03\n\rTypeSpecProto\x12\x41\n\x0ftype_spec_class\x18\x01 \x01(\x0e\x32(.tensorboard.TypeSpecProto.TypeSpecClass\x12\x30\n\ntype_state\x18\x02 \x01(\x0b\x32\x1c.tensorboard.StructuredValue\x12\x1c\n\x14type_spec_class_name\x18\x03 \x01(\t\x12\x1b\n\x13num_flat_components\x18\x04 \x01(\x05\"\xb8\x02\n\rTypeSpecClass\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x16\n\x12SPARSE_TENSOR_SPEC\x10\x01\x12\x17\n\x13INDEXED_SLICES_SPEC\x10\x02\x12\x16\n\x12RAGGED_TENSOR_SPEC\x10\x03\x12\x15\n\x11TENSOR_ARRAY_SPEC\x10\x04\x12\x15\n\x11\x44\x41TA_DATASET_SPEC\x10\x05\x12\x16\n\x12\x44\x41TA_ITERATOR_SPEC\x10\x06\x12\x11\n\rOPTIONAL_SPEC\x10\x07\x12\x14\n\x10PER_REPLICA_SPEC\x10\x08\x12\x11\n\rVARIABLE_SPEC\x10\t\x12\x16\n\x12ROW_PARTITION_SPEC\x10\n\x12\x18\n\x14REGISTERED_TYPE_SPEC\x10\x0c\x12\x17\n\x13\x45XTENSION_TYPE_SPEC\x10\r\"\x04\x08\x0b\x10\x0b\x42WZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_protob\x06proto3')



_STRUCTUREDVALUE = DESCRIPTOR.message_types_by_name['StructuredValue']
_NONEVALUE = DESCRIPTOR.message_types_by_name['NoneValue']
_LISTVALUE = DESCRIPTOR.message_types_by_name['ListValue']
_TUPLEVALUE = DESCRIPTOR.message_types_by_name['TupleValue']
_DICTVALUE = DESCRIPTOR.message_types_by_name['DictValue']
_DICTVALUE_FIELDSENTRY = _DICTVALUE.nested_types_by_name['FieldsEntry']
_PAIRVALUE = DESCRIPTOR.message_types_by_name['PairValue']
_NAMEDTUPLEVALUE = DESCRIPTOR.message_types_by_name['NamedTupleValue']
_TENSORSPECPROTO = DESCRIPTOR.message_types_by_name['TensorSpecProto']
_BOUNDEDTENSORSPECPROTO = DESCRIPTOR.message_types_by_name['BoundedTensorSpecProto']
_TYPESPECPROTO = DESCRIPTOR.message_types_by_name['TypeSpecProto']
_TYPESPECPROTO_TYPESPECCLASS = _TYPESPECPROTO.enum_types_by_name['TypeSpecClass']
StructuredValue = _reflection.GeneratedProtocolMessageType('StructuredValue', (_message.Message,), {
  'DESCRIPTOR' : _STRUCTUREDVALUE,
  '__module__' : 'tensorboard.compat.proto.struct_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.StructuredValue)
  })
_sym_db.RegisterMessage(StructuredValue)

NoneValue = _reflection.GeneratedProtocolMessageType('NoneValue', (_message.Message,), {
  'DESCRIPTOR' : _NONEVALUE,
  '__module__' : 'tensorboard.compat.proto.struct_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.NoneValue)
  })
_sym_db.RegisterMessage(NoneValue)

ListValue = _reflection.GeneratedProtocolMessageType('ListValue', (_message.Message,), {
  'DESCRIPTOR' : _LISTVALUE,
  '__module__' : 'tensorboard.compat.proto.struct_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.ListValue)
  })
_sym_db.RegisterMessage(ListValue)

TupleValue = _reflection.GeneratedProtocolMessageType('TupleValue', (_message.Message,), {
  'DESCRIPTOR' : _TUPLEVALUE,
  '__module__' : 'tensorboard.compat.proto.struct_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.TupleValue)
  })
_sym_db.RegisterMessage(TupleValue)

DictValue = _reflection.GeneratedProtocolMessageType('DictValue', (_message.Message,), {

  'FieldsEntry' : _reflection.GeneratedProtocolMessageType('FieldsEntry', (_message.Message,), {
    'DESCRIPTOR' : _DICTVALUE_FIELDSENTRY,
    '__module__' : 'tensorboard.compat.proto.struct_pb2'
    # @@protoc_insertion_point(class_scope:tensorboard.DictValue.FieldsEntry)
    })
  ,
  'DESCRIPTOR' : _DICTVALUE,
  '__module__' : 'tensorboard.compat.proto.struct_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.DictValue)
  })
_sym_db.RegisterMessage(DictValue)
_sym_db.RegisterMessage(DictValue.FieldsEntry)

PairValue = _reflection.GeneratedProtocolMessageType('PairValue', (_message.Message,), {
  'DESCRIPTOR' : _PAIRVALUE,
  '__module__' : 'tensorboard.compat.proto.struct_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.PairValue)
  })
_sym_db.RegisterMessage(PairValue)

NamedTupleValue = _reflection.GeneratedProtocolMessageType('NamedTupleValue', (_message.Message,), {
  'DESCRIPTOR' : _NAMEDTUPLEVALUE,
  '__module__' : 'tensorboard.compat.proto.struct_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.NamedTupleValue)
  })
_sym_db.RegisterMessage(NamedTupleValue)

TensorSpecProto = _reflection.GeneratedProtocolMessageType('TensorSpecProto', (_message.Message,), {
  'DESCRIPTOR' : _TENSORSPECPROTO,
  '__module__' : 'tensorboard.compat.proto.struct_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.TensorSpecProto)
  })
_sym_db.RegisterMessage(TensorSpecProto)

BoundedTensorSpecProto = _reflection.GeneratedProtocolMessageType('BoundedTensorSpecProto', (_message.Message,), {
  'DESCRIPTOR' : _BOUNDEDTENSORSPECPROTO,
  '__module__' : 'tensorboard.compat.proto.struct_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.BoundedTensorSpecProto)
  })
_sym_db.RegisterMessage(BoundedTensorSpecProto)

TypeSpecProto = _reflection.GeneratedProtocolMessageType('TypeSpecProto', (_message.Message,), {
  'DESCRIPTOR' : _TYPESPECPROTO,
  '__module__' : 'tensorboard.compat.proto.struct_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.TypeSpecProto)
  })
_sym_db.RegisterMessage(TypeSpecProto)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto'
  _DICTVALUE_FIELDSENTRY._options = None
  _DICTVALUE_FIELDSENTRY._serialized_options = b'8\001'
  _STRUCTUREDVALUE._serialized_start=177
  _STRUCTUREDVALUE._serialized_end=942
  _NONEVALUE._serialized_start=944
  _NONEVALUE._serialized_end=955
  _LISTVALUE._serialized_start=957
  _LISTVALUE._serialized_end=1014
  _TUPLEVALUE._serialized_start=1016
  _TUPLEVALUE._serialized_end=1074
  _DICTVALUE._serialized_start=1077
  _DICTVALUE._serialized_end=1217
  _DICTVALUE_FIELDSENTRY._serialized_start=1142
  _DICTVALUE_FIELDSENTRY._serialized_end=1217
  _PAIRVALUE._serialized_start=1219
  _PAIRVALUE._serialized_end=1288
  _NAMEDTUPLEVALUE._serialized_start=1290
  _NAMEDTUPLEVALUE._serialized_end=1361
  _TENSORSPECPROTO._serialized_start=1363
  _TENSORSPECPROTO._serialized_end=1478
  _BOUNDEDTENSORSPECPROTO._serialized_start=1481
  _BOUNDEDTENSORSPECPROTO._serialized_end=1689
  _TYPESPECPROTO._serialized_start=1692
  _TYPESPECPROTO._serialized_end=2198
  _TYPESPECPROTO_TYPESPECCLASS._serialized_start=1886
  _TYPESPECPROTO_TYPESPECCLASS._serialized_end=2198
# @@protoc_insertion_point(module_scope)