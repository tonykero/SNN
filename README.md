# **Simple Neural Networks**

SNN is a lightweight C++ library which focus on Artificial Neural Networks,
Aiming to be intuitive, fast & complete.


## Table of Contents

* Why ?
* Status
    * Core Features
* Planned implementations
* Compiling
* Documentation
* About
* License

---

## Why ?

As a person who gets interested in a lot of things in Computer Science,
i always try to understand principles and then code the correspondings algorithms.
But the thing is that sometimes i get limited by libraries, or i am either disapointed,
by the complexity and the lack of features of some.

So i decided to make my own library, for artificial neural networks,
Actually **creating a 2 Layers-Deep FeedForward Neural Network and training it**
**with a Genetic Algorithm using my library, is about writing 7 lines of code.**

---

## Status

This library is still **in development**
However some core features work.


### Core Features:
```
- Net                   (done)
- Activations Functions (done)
- Layer/Neuron/Link     (done)
- FFNet                 (done)
- Genetic Algorithm     (done)
- Backpropagation
- Recurrent
```
---
## Planned implementations:

* Neural Network Types:
    * Self-Organizing Maps
    * FeedForward
    * Convolutional
    * Recurrent
    
* Activations Functions:
    * Linear
    * Step
    * Sigmoid
    * Hyperbolic Tangent
    * Rectified Linear Unit

* Training Algorithms:
    * Backpropagation
    * Resilient Propagation
    * Genetic Algorithm

* Model Selection:
    * Pruning
    * Regularization & Dropout
    * Grid search (brute force)
---
## Compiling

### Requirements

- CMake 2.8
- g++ or clang++ or msvc

You can use CMake to compile SNN.
the script supports MSVC, Clang & GCC

3 options are defined:
```
BUILD_SHARED    (default: ON)
BUILD_EXAMPLES  (default: OFF)
DEBUG           (default: OFF)
```

First clone the repository:
```
git clone https://github.com/tonykero/SNN.git SNN
cd SNN/build
```
The following command will configure the library as a RELEASE SHARED library and without examples
and the generator choosen will be Visual Studio if under Windows, or Unix Makefiles if under Linux.
You can change this with -G option.
```
cmake ..
```
To build the library and examples:
```
cmake -DBUILD_EXAMPLES=ON ..
```
To build the library as STATIC library:
```
cmake -DBUILD_SHARED=OFF ..
```

And finally run the build
```
cmake --build .
```

---

## Documentation

Documentation is planned.

---

## About

SNN is developed by me :] as a french autodidact/hobbyist nerd/no-life Computer Science lover.

---

## License

This library is licensed under [GNU GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)