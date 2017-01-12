## SeetaFace Engine

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

### Description

**SeetaFace Engine** is an open source C++ face recognition engine, which can run on CPU with no third-party dependence. It contains three key parts, i.e., **SeetaFace Detection**, **SeetaFace Alignment** and **SeetaFace Identification**, which are necessary and sufficient for building a real-world face recognition applicaiton system.

* SeetaFace Detection implements a funnel-structured (FuSt) cascade schema for real-time multi-view face detection, which achieves a good trade-off between detection accuracy and speed. State of the art accuracy can be achieved on the public dataset [FDDB](http://vis-www.cs.umass.edu/fddb/) in high speed. See [SeetaFace Detection](https://github.com/seetaface/SeetaFaceEngine/tree/master/FaceDetection) for more details.

* SeetaFace Alignment cascades several stacked auto-encoder networks to detect landmarks in real time (more than 200 fps on a single I7 desktop CPU), and achieves the state-of-the-art accuracy at least on some public datasets [AFLW](http://lrs.icg.tugraz.at/research/aflw/). See [SeetaFace Alignment](https://github.com/seetaface/SeetaFaceEngine/tree/master/FaceAlignment) for more details.

* SeetaFace Identification is a modification of AlexNet CNN for face recognition, with better performance in terms of both accuracy (97.1% on [LFW] (http://vis-www.cs.umass.edu/lfw/) and speed (about 120ms on a single I7 desktop CPU). See [SeetaFace Identification](https://github.com/seetaface/SeetaFaceEngine/tree/master/FaceIdentification) for more details.

This face recognition engine is developed by Visual Information Processing and Learning (VIPL) group, Institute of Computing Technology, Chinese Academy of Sciences. The codes are written in C++ without dependence on any 3rd-party libraries. The open source is now released under BSD-2 license (see [LICENSE](LICENSE) for details), which means the codes can be used freely for both acedemic purpose and industrial products.

### Contact Info

If you have any problem on SeetaFace Engine, please contact us by sending email to SeetaFace@vipl.ict.ac.cn.


### Other Documentation

* [SeetaFace Detection](./FaceDetection/README.md)
* [SeetaFace Alignment](./FaceAlignment/README.md)
* [SeetaFace Identification](./FaceIdentification/README.md)

## Setup

### Prerequisites

+ Linux or OSX
+ rar(If you haven't installed it yet: sudo apt-get install rar)
+ cmake-2.8.4 +
+ OpenCV2(No need if you build without the example)

### Make a choice

+ Set option **on** in CMakeLists.txt to build with the example(Need OpenCV2).

+ Set option **off** in CMakeLists.txt to build the shared library only(No need of OpenCV).

### Getting Started

    cd SeetaFaceCool
    rar e model/model.part1.rar model/
    mkdir build
    cd build
    cmake -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF ..
    make

Then an executable file is generated in SeetaFaceCool/build/ and three shared
libraries(.so) are generated in SeetaFaceCool/lib/

To run the seetaface_example:

    cd SeetaFaceCool
    ./build/seetaface_example Data/tangwei_gallery.jpg Data/tangwei_probe.jpg

Modify the code in seetaface_example.cpp and re-execute the above process to
realize your own function.
