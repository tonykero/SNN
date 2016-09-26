#pragma once

#include <cmath>

namespace snn
{
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
        return tanh(fout);
    }
}