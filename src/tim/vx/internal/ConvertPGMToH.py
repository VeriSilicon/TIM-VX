#!/usr/bin/env python

import sys
import os
import re


def checkFile(path):
    return os.path.isfile(path)


def toHex(number):
    return "0x%08X" % number


def usage():
    print("ConvertPGMToH: convert VXC PGM file to C header file")
    print("Usage: ConvertPGMToH [-h] [-a] -i input_file -o output_file")
    print("Options and arguments:")
    print("    -h       :show this usage")
    print("    -a       :append to the output_file")
    print("    -i input_file  :specify the .gcPGM file which include VXC shader binary")
    print("    -o output_file :specify the .h file to save converted data")


def ConverBinToH(source, target):
    global append
    mmtime = os.path.getmtime(source)

    if checkFile(target):
        if os.path.getmtime(target) > mmtime:
            print("%s is already the latest one (nothing changed)" % target)
            return

    with open(source, "rb") as fin:
        buf = fin.read()
    #os.remove(target)
    bytes = bytearray(buf)

    fout = None
    if (append): 
        fout = open(target, "a")
    else:
        fout = open(target, "w")

    if not (fout):
        print("ERROR: failed to open file: %s" % (target))

    #fout.write(datetime.datetime.now().strftime("/*Auto created on %Y-%m-%d %H:%M:%S*/\n"))
    name = os.path.basename(source).replace(".", "_", 1)
    name = re.sub("_all.gcPGM", "", name);
    if (append == 0):
        fout.write("#ifndef __%s_H__\n#define __%s_H__\n\n" % (name.upper(), name.upper()))

    fout.write("static uint8_t vxcBin%s[] = {" % (name))
    i = 0
    for c in bytes:
        if i % 8 == 0:
            fout.write("\n  ")
        fout.write("0x%02X, " % c)
        i += 1

    fout.write("\n};\n")
    fout.write("#define VXC_BIN_%s_LEN 0x%08XU\n" % (name.upper(), i))

    if (append == 0):
        fout.write("\n#endif\n\n");
    fout.close()
    #print("%s is generated/refreshed" % target)
    return

def main():
    global append
    args = sys.argv
    argc = len(args)
    target_path = "."
    source = ""
    if argc <= 1:
        usage()
        return
    i = 1
    while i < argc:
        if args[i] == "-h":
            usage()
            return
        if args[i] == "-a":
            append = 1
        elif args[i] == "-o":
            i += 1
            target_path = args[i]
        elif args[i] == "-i":
            i += 1
            source = args[i]
        else:
            print("unknown parameter: " + args[i])
            usage()
            return
        i += 1

    if source == "":
        print("Please specify .c/cpp/h/hpp file")
        return
    else:
        if not checkFile(source):
            print("Cannot find the file %s" % source)
            return

    ConverBinToH(source, target_path)
    return


append = 0
if __name__=='__main__':
    main()
