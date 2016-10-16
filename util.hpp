#pragma once

#include <cmath>

namespace snn
{
    /*Activations Functions*/
    inline float FUN_LINEAR(float fout)
    {
        return fout;
    }

    inline float FUN_STEP(float fout);
    {
        return (fout > 0 ? 1 : 0);
    }

    inline float FUN_SIGMOID(float fout);
    {
        return (1/(1+exp(-fout)));
    }

    inline float FUN_TANH(float fout);
    {
        //or just tanh(fout)
        float e = exp(2*fout);
        return (e-1)/(e+1);
    }

    inline float FUN_RELU(float fout)
    {
        //or just max(0, fout); but this formula rocks
        return (abs(fout)+fout)/2; 
    }
}