#define TENSORLOGICAL(name, lgc_op, lgc_op2) \
__kernel void logical_##name##_I8toI8( \
    __read_only image2d_array_t   input, \
    __read_only image2d_array_t   input1, \
    __write_only image2d_array_t  output) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    int4 src0; \
    int4 src1; \
    READ_IMAGEI_2DARRAY(src0, input, coord); \
    READ_IMAGEI_2DARRAY(src1, input1, coord); \
    int4 dst  = (lgc_op2(src0))lgc_op(lgc_op2(src1)); \
    dst.x = dst.x & 1; \
    write_imagei(output, coord, dst); \
}

TENSORLOGICAL(or,  ||, )
TENSORLOGICAL(and, &&, )
TENSORLOGICAL(xor, ^,  !!)


#define TENSORLOGICAL_2D(name, lgc_op, lgc_op2) \
__kernel void logical_##name##_I8toI8_2D( \
    __read_only image2d_t   input, \
    __read_only image2d_t   input1, \
    __write_only image2d_t  output) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
    int4 src0 = read_imagei(input, coord); \
    int4 src1 = read_imagei(input1, coord); \
    int4 dst  = (lgc_op2(src0))lgc_op(lgc_op2(src1)); \
    dst.x = dst.x & 1; \
    write_imagei(output, coord, dst); \
}

TENSORLOGICAL_2D(or,  ||, )
TENSORLOGICAL_2D(and, &&, )
TENSORLOGICAL_2D(xor, ^,  !!)
