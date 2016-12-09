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
#include "net.hpp"
#include "util.hpp"
#include "datasets.hpp"

namespace snn
{
    enum BackpropType: unsigned int
    {
        ONLINE = 1,
        BATCH
    };

    class BackpropTrainer
    {
        public:

            BackpropTrainer(unsigned int _backpropType, float _learningRate, float _momentum);
            ~BackpropTrainer();

            void setOutputDerivative(std::function<float(float)> _actFunDerivative);
            void setHiddenDerivative(std::function<float(float)> _actFunDerivative);

            void train(Net* _net, unsigned int _epochs, std::vector<Dataset> _datasets);
        
        private:
            unsigned int backpropType_m;
            float learningRate_m;
            float momentum_m;
            
            std::function<float(float)> outputDer_m = util::sigmoidDerivative;
            std::function<float(float)> hiddenDer_m = util::sigmoidDerivative;
            
            void trainOnline(Net* _net, unsigned int _epochs, std::vector<Dataset> _datasets);
            void trainBatch(Net* _net, unsigned int _epochs, std::vector<Dataset> _datasets);
    };
}
