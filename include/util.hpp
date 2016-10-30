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

#include "Config.hpp"
#include <cmath>
#include <vector>

namespace snn
{
    namespace util
    {
        /*Activations Functions*/
    inline float linear(float fout)
    {
        return fout;
    }

    inline float step(float fout)
    {
        return (fout > 0.0f ? 1.0f : 0.0f);
    }

    inline float sigmoid(float fout)
    {
        return (1.0f/(1.0f+exp(-fout)));
    }

    inline float tanh(float fout)
    {
        //or just tanh(fout)
        float e = exp(2.0f*fout);
        return (e-1.0f)/(e+1.0f);
    }

    inline float relu(float fout)
    {
        //or just max(0, fout); but this formula rocks
        return (std::abs(fout)+fout)/2.0f; 
    }

    }

    class Dataset
    {
        public:
            Dataset(std::vector<float> _inputs, std::vector<float> _outputs)
            {
                inputs = _inputs;
                outputs = _outputs;
            }

            ~Dataset() {};

            std::vector<float> inputs;
            std::vector<float> outputs;
    };
    
}