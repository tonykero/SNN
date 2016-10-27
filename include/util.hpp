/*********************************************************************
*Copyright (C) 2016  Antoine Karcher				                 *
*								                                     *
*This program is free software: you can redistribute it and/or modify*
*it under the terms of the GNU General Public License as published by*
*the Free Software Foundation, either version 3 of the License, or   *
*(at your option) any later version.				                 *
*								                                     *
*This program is distributed in the hope that it will be useful,     *
*but WITHOUT ANY WARRANTY; without even the implied warranty of	     *
*MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the       *
*GNU General Public License for more details.			             *
*                                                                    *
*You should have received a copy of the GNU General Public License   *
*along with this program. If not, see <http://www.gnu.org/licenses/>.*
*********************************************************************/

#pragma once

#include <cmath>

namespace snn
{
    /*Activations Functions*/
    inline float FUN_LINEAR(float fout)
    {
        return fout;
    }

    inline float FUN_STEP(float fout)
    {
        return (fout >= 0.0f ? 1.0f : 0.0f);
    }

    inline float FUN_SIGMOID(float fout)
    {
        return (1/(1+exp(-fout)));
    }

    inline float FUN_TANH(float fout)
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

    /*Derivatives*/
    inline float DER_SIGMOID(float fout)
    {
        float sgd = FUN_SIGMOID(fout);
        return (sgd*(1-sgd));
    }

    inline float DER_TANH(float fout)
    {
        float tanh = FUN_TANH(fout);
        return 1-(tanh*tanh);
    }

    inline float DER_RELU(float fout)
    {
        float out = 0.0f;
        if(fout != 0.0f)
            out =  (abs(fout)+fout)/(2*fout);
        return out;
    }
}