#pragma OPENCL EXTENSION CL_VIV_asm : enable
#pragma OPENCL EXTENSION cl_viv_vx_extension : enable

#define readImage2DArray(Dest, Image, Coord)         \
    do {                                                       \
       int8 desc;                                              \
       _viv_asm(COPY, desc, Image, sizeof(desc));              \
       _viv_asm(CLAMP0MAX, (Coord).w, (Coord).z, desc.s5 - 1); \
       int baseAddr =  (int)(Coord).w * desc.s4 + desc.s0;     \
       _viv_asm(MOV, (Coord).w, baseAddr);                     \
       _viv_asm(IMAGE_READ_3D, Dest, Image, (Coord).xyww);     \
    } while (0)

#define writeImage2DArray(Image, Coord, Color)                 \
    do {                                                       \
       int8 desc;                                              \
       _viv_asm(COPY, desc, Image, sizeof(desc));              \
       _viv_asm(CLAMP0MAX, (Coord).w, (Coord).z, desc.s5 - 1); \
       int baseAddr =  (int)(Coord).w * desc.s4 + desc.s0;     \
       _viv_asm(MOV, (Coord).w, baseAddr);                     \
       _viv_asm(IMAGE_WRITE_3D, Color, Image, (Coord).xyww);   \
    } while (0)

#define readImage(Dest, Image, Coord)               \
    do {                                            \
       _viv_asm(IMAGE_READ, Dest, Image, Coord);    \
    } while (0)

#define writeImage(Image, Coord, Color)             \
    do {                                            \
       _viv_asm(IMAGE_WRITE, Color, Image, Coord);   \
    } while (0)
