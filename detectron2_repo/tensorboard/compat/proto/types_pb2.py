# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorboard/compat/proto/types.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n$tensorboard/compat/proto/types.proto\x12\x0btensorboard\":\n\x0fSerializedDType\x12\'\n\x08\x64\x61tatype\x18\x01 \x01(\x0e\x32\x15.tensorboard.DataType*\x86\x07\n\x08\x44\x61taType\x12\x0e\n\nDT_INVALID\x10\x00\x12\x0c\n\x08\x44T_FLOAT\x10\x01\x12\r\n\tDT_DOUBLE\x10\x02\x12\x0c\n\x08\x44T_INT32\x10\x03\x12\x0c\n\x08\x44T_UINT8\x10\x04\x12\x0c\n\x08\x44T_INT16\x10\x05\x12\x0b\n\x07\x44T_INT8\x10\x06\x12\r\n\tDT_STRING\x10\x07\x12\x10\n\x0c\x44T_COMPLEX64\x10\x08\x12\x0c\n\x08\x44T_INT64\x10\t\x12\x0b\n\x07\x44T_BOOL\x10\n\x12\x0c\n\x08\x44T_QINT8\x10\x0b\x12\r\n\tDT_QUINT8\x10\x0c\x12\r\n\tDT_QINT32\x10\r\x12\x0f\n\x0b\x44T_BFLOAT16\x10\x0e\x12\r\n\tDT_QINT16\x10\x0f\x12\x0e\n\nDT_QUINT16\x10\x10\x12\r\n\tDT_UINT16\x10\x11\x12\x11\n\rDT_COMPLEX128\x10\x12\x12\x0b\n\x07\x44T_HALF\x10\x13\x12\x0f\n\x0b\x44T_RESOURCE\x10\x14\x12\x0e\n\nDT_VARIANT\x10\x15\x12\r\n\tDT_UINT32\x10\x16\x12\r\n\tDT_UINT64\x10\x17\x12\x12\n\x0e\x44T_FLOAT8_E5M2\x10\x18\x12\x14\n\x10\x44T_FLOAT8_E4M3FN\x10\x19\x12\x10\n\x0c\x44T_FLOAT_REF\x10\x65\x12\x11\n\rDT_DOUBLE_REF\x10\x66\x12\x10\n\x0c\x44T_INT32_REF\x10g\x12\x10\n\x0c\x44T_UINT8_REF\x10h\x12\x10\n\x0c\x44T_INT16_REF\x10i\x12\x0f\n\x0b\x44T_INT8_REF\x10j\x12\x11\n\rDT_STRING_REF\x10k\x12\x14\n\x10\x44T_COMPLEX64_REF\x10l\x12\x10\n\x0c\x44T_INT64_REF\x10m\x12\x0f\n\x0b\x44T_BOOL_REF\x10n\x12\x10\n\x0c\x44T_QINT8_REF\x10o\x12\x11\n\rDT_QUINT8_REF\x10p\x12\x11\n\rDT_QINT32_REF\x10q\x12\x13\n\x0f\x44T_BFLOAT16_REF\x10r\x12\x11\n\rDT_QINT16_REF\x10s\x12\x12\n\x0e\x44T_QUINT16_REF\x10t\x12\x11\n\rDT_UINT16_REF\x10u\x12\x15\n\x11\x44T_COMPLEX128_REF\x10v\x12\x0f\n\x0b\x44T_HALF_REF\x10w\x12\x13\n\x0f\x44T_RESOURCE_REF\x10x\x12\x12\n\x0e\x44T_VARIANT_REF\x10y\x12\x11\n\rDT_UINT32_REF\x10z\x12\x11\n\rDT_UINT64_REF\x10{\x12\x16\n\x12\x44T_FLOAT8_E5M2_REF\x10|\x12\x18\n\x14\x44T_FLOAT8_E4M3FN_REF\x10}Bz\n\x18org.tensorflow.frameworkB\x0bTypesProtosP\x01ZLgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/types_go_proto\xf8\x01\x01\x62\x06proto3')

_DATATYPE = DESCRIPTOR.enum_types_by_name['DataType']
DataType = enum_type_wrapper.EnumTypeWrapper(_DATATYPE)
DT_INVALID = 0
DT_FLOAT = 1
DT_DOUBLE = 2
DT_INT32 = 3
DT_UINT8 = 4
DT_INT16 = 5
DT_INT8 = 6
DT_STRING = 7
DT_COMPLEX64 = 8
DT_INT64 = 9
DT_BOOL = 10
DT_QINT8 = 11
DT_QUINT8 = 12
DT_QINT32 = 13
DT_BFLOAT16 = 14
DT_QINT16 = 15
DT_QUINT16 = 16
DT_UINT16 = 17
DT_COMPLEX128 = 18
DT_HALF = 19
DT_RESOURCE = 20
DT_VARIANT = 21
DT_UINT32 = 22
DT_UINT64 = 23
DT_FLOAT8_E5M2 = 24
DT_FLOAT8_E4M3FN = 25
DT_FLOAT_REF = 101
DT_DOUBLE_REF = 102
DT_INT32_REF = 103
DT_UINT8_REF = 104
DT_INT16_REF = 105
DT_INT8_REF = 106
DT_STRING_REF = 107
DT_COMPLEX64_REF = 108
DT_INT64_REF = 109
DT_BOOL_REF = 110
DT_QINT8_REF = 111
DT_QUINT8_REF = 112
DT_QINT32_REF = 113
DT_BFLOAT16_REF = 114
DT_QINT16_REF = 115
DT_QUINT16_REF = 116
DT_UINT16_REF = 117
DT_COMPLEX128_REF = 118
DT_HALF_REF = 119
DT_RESOURCE_REF = 120
DT_VARIANT_REF = 121
DT_UINT32_REF = 122
DT_UINT64_REF = 123
DT_FLOAT8_E5M2_REF = 124
DT_FLOAT8_E4M3FN_REF = 125


_SERIALIZEDDTYPE = DESCRIPTOR.message_types_by_name['SerializedDType']
SerializedDType = _reflection.GeneratedProtocolMessageType('SerializedDType', (_message.Message,), {
  'DESCRIPTOR' : _SERIALIZEDDTYPE,
  '__module__' : 'tensorboard.compat.proto.types_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SerializedDType)
  })
_sym_db.RegisterMessage(SerializedDType)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\030org.tensorflow.frameworkB\013TypesProtosP\001ZLgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/types_go_proto\370\001\001'
  _DATATYPE._serialized_start=114
  _DATATYPE._serialized_end=1016
  _SERIALIZEDDTYPE._serialized_start=53
  _SERIALIZEDDTYPE._serialized_end=111
# @@protoc_insertion_point(module_scope)