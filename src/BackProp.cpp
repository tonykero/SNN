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

#include "../include/BackProp.hpp"

#include <iostream>

using namespace snn;

BackProp::BackProp(FFNet* net, std::function<float(float)> hidFun_derivative = DER_SIGMOID, std::function<float(float)> outFun_derivative = DER_SIGMOID)
{
    m_pnet = net;

}

BackProp::~BackProp()
{
    //delete m_pnet;
}

void BackProp::train(std::vector<Dataset> ds, float learningRate, float momentum, unsigned int maxIterations)
{
    //online Backpropagation:
    //for each dataset
    //  feed
    //  compute outputs deltas
    //  compute interior deltas
    //  for each link
    //      LINK.WEIGHT += LearningRate*LINK.B.DELTA*LINK.A.OUTPUT + MOMENTUM*LINK.PREVIOUS_WEIGHT

    for(unsigned int i = 0; i < maxIterations; i++)
    {
        for(unsigned int j = 0; j < ds.size(); j++)
        {
            std::vector<float> outputs = m_pnet->feed(ds[j].inputs); //sets all internal values stuff (outputs, sums)

            /*bae*/
            m_pnet->computeDeltas(ds[j].outputs, outputs);

            //updateWeights
            std::vector<Link> links = m_pnet->getLinks();

			//std::cout << "\n-------------------\n";
            for(unsigned int k = 0; k < links.size(); k++)
            {
                //Link: A->B
                float delta_B = m_pnet->getNeurons()[links[k].b].delta;
                float output_A = m_pnet->getNeurons()[links[k].a].sum;

                float tmpC = links[k].previousChange;
                links[k].weight += learningRate*delta_B*output_A + momentum*links[k].previousChange;
                links[k].previousChange = learningRate*delta_B*output_A + momentum*links[k].previousChange;
				//std::cout << "\nchange: " << links[k].previousChange << "\n";

            }

			m_pnet->setLinks(links);

        }
    }
}