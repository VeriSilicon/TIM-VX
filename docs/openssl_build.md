OpenSSL doesn't have cmake project by default, so we use [ssl.cmake](https://github.com/viaduck/openssl-cmake) to integrate OpenSSL build with cmake.

# Build openssl from downloaded source archive
```bash
mkdir build && cd build
cmake .. -DTIM_VX_ENABLE_TENSOR_CACHE=ON -DBUILD_OPENSSL=ON
```

# Build openssl with local source repo
```bash
git clone git@github.com:openssl/openssl.git ${local_ssl_repo}
# checkout to 1.1.1t or other release tag
# build tim-vx with local ssl source repo
mkdir build && cd build
cmake .. -DTIM_VX_ENABLE_TENSOR_CACHE=ON -DBUILD_OPENSSL=ON -DLOCAL_BUILD=ON -DLOCAL_OPENSSL=${local_ssl_repo}
```

# Build openssl with android-ndk
```bash
cmake .. -DTIM_VX_ENABLE_TENSOR_CACHE=ON -DBUILD_OPENSSL=ON -DLOCAL_BUILD=ON -DLOCAL_OPENSSL=${local_ssl_repo} -DCMAKE_TOOLCHAIN_FILE=${ndk_root}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DCROSS_ANDROID=ON -DEXTERNAL_VIV_SDK=${ANDROID_OVX_DRV_PREBUILD}
# change ANDROID_ABI accroding to target platform
```

# Build openssl with mirrored ssl.cmake
If [ssl.cmake](https://github.com/viaduck/openssl-cmake) not always accessable from your originization, you can mirror it and replace the download url by append **-DOPENSSL_CMAKE_URL=<ssl.cmake mirror>** to cmake command line.

# Deployment with OpenSSL libraries
All libraries can be installed in single place by
```bash
make install # will install all required library(libtim-vx, libssl, libcrypto) to <build>/install/lib/
```

# TODO: General cross build with openssl not support yet