#pragma once
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

#include <vector>
#include <functional>
#include <string>

#include "util.hpp"


namespace snn
{

    enum NType: unsigned int
    {

        INPUT = 1,
        HIDDEN,
        OUTPUT,
        BIAS,
        CONTEXT
        
    };

	struct Neuron;
	class Net;

    struct Link
    {
        unsigned int a;
        unsigned int b;

        float weight = 1.0f;
    };

    struct Neuron
    {
        Net* parent;
        float output = 0.0f;

        unsigned int type; //INPUT, HIDDEN, OUTPUT, BIAS
		unsigned int id; //unique identifier
        
        float compute();

        friend class Net;
    };

    class Net
    {
        public:
            Net(std::vector<Neuron> neurons, std::vector<Link> links);

            ~Net();

            std::vector<float> feed(std::vector<float> inputs);

            void setOutputFunction(std::function<float(float)> actFun);
            void setHiddenFunction(std::function<float(float)> actFun);

            std::function<float(float)> getOutputFunction();
            std::function<float(float)> getHiddenFunction();

            unsigned int getSize();

            void randWeights();
            void setLinks(std::vector<Link>);
            std::vector<Link> getLinks();

        private:

            std::vector<Neuron> m_neurons;

            unsigned int m_neuronsCount = 0;

            std::vector<Link> m_links;

			unsigned int m_linksCount = 0;

            std::vector<float> m_inputs;

            std::function<float(float)> m_outFun = snn::FUN_SIGMOID;
            std::function<float(float)> m_hidFun = snn::FUN_SIGMOID;

            friend struct Neuron;
    };

}

