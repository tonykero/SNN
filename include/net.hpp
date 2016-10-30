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
#include "util.hpp"

#include <vector>
#include <functional>




namespace snn
{

    enum NType: unsigned int
    {

        INPUT = 1,
        HIDDEN,
        OUTPUT,
        BIAS,
        
    };

	struct Neuron;
	class Net;

    struct Layer
    {
        Layer(unsigned int _neuronsCount, unsigned int _neuronsType, unsigned int _biasCount = 0);
        std::vector<Neuron> neurons;//neurons initialized without id
        unsigned int type;
    };

    struct Link
    {
        unsigned int a;
        unsigned int b;

        float weight = 1.0f;
    };

    struct Neuron
    {
        Net* parent;									  //defined in Net(...)
        
		float output = 0.0f;
		float sum = 0.0f;

        unsigned int type; //INPUT, HIDDEN, OUTPUT, BIAS	
		unsigned int id; //unique identifier				defined in Net(...)
        
        float compute(std::vector<float> _inputs);
    };

    class Net
    {
        public:
			Net(std::vector<Neuron> _neurons, std::vector<Link> _links);

            virtual ~Net();

            std::vector<float> feed(std::vector<float> _inputs);

			float computeMSE(std::vector<float> _inputs, std::vector<float> _outputs);
			float computeGlobalMSE(std::vector<Dataset> _ds);

            void setOutputFunction(std::function<float(float)> _actFun);
            void setHiddenFunction(std::function<float(float)> _actFun);

            std::function<float(float)> getOutputFunction();
            std::function<float(float)> getHiddenFunction();

            unsigned int getSize();

            void randWeights(float, float);
            void setLinks(std::vector<Link>);
            std::vector<Link> getLinks();

			std::vector<Neuron> getNeurons();

        private:

            std::vector<float> inputs_m;

            std::function<float(float)> outFun_m = snn::util::sigmoid;
            std::function<float(float)> hidFun_m = snn::util::sigmoid;
        
        protected:
			Net();

            std::vector<Neuron> neurons_m;

            unsigned int neuronsCount_m = 0;

            std::vector<Link> links_m;

			unsigned int linksCount_m = 0;
        
        friend struct Neuron;
    };

}

