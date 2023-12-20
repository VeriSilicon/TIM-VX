# How to prebuild VXC binary file for shader scripts in ovxlib?

Generally shader scripts will be compiled into VXC binary file on app run time. But this compilation will cost a lot of time. If we prebuild VXC binary file, this time will be saved.

## prepare

Full driver source is needed. You MUST build driver firstly, and make sure `vcCompiler` is build properly.

## build

```
cd tim/vx/internal/
./ovxlib_bin_build.sh <VIVANTE SDK path> <vcCompiler path> <ovxlib path(for example: src/tim/vx/internal/)> <config file path(for example: viv_vip8000nanosi+_pid0x9F.config)>
```

After build, you will find many `.h` files in `<ovxlib path>/include/libnnext/vx_bin`. These `.h` files will be compiled into libtim-vx.so when build TIM-VX.
