# **Simple Neural Networks**

SNN is a lightweight C++ library which focus on artificial neural Networks,
Aiming to be intuitive, fast & complete.

## Why ?

As a person who gets interested in a lot of things in Computer Science,
i always try to understand principles and then code the correspondings algorithms.
But the thing is that sometimes i get limited by libraries, or i am either disapointed,
by the complexity and the lack of features of some.

So i decided to make my own library, for artificial neural networks,
Actually **creating a 2 Layers-Deep FeedForward Neural Network and training it**
**with a Genetic Algorithm using my library, is about writing 7 lines of code.**

## Status

This library is still **in development**
However some core features work.

## Compiling

At this moment no proper compilation process is implemented.
However you will find batch scripts in bin/ to compile examples
under Windows with g++.

CMake build script is planned.

### Core Features:
```
- Net                   (done)
- Activations Functions (done)
- Layer/Neuron/Link     (done)
- FFNet                 (done)
- Genetic Algorithm     (done)
- Backpropagation
- SOM
- Convolutional
- Recurrent
```

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

## Documentation

Documentation is planned.

## About

SNN is developed by me :] as a french autodidact/hobbyist nerd

## License

This library is licensed under [GNU GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)