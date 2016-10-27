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

#include "../include/ffnet.hpp"
#include <iostream>

using namespace snn;

FFNet::FFNet(std::vector<Layer> layers)
{
    m_layersCount = layers.size();
	m_layers = layers;

    std::vector<Neuron> neurons = layers2vector();

    std::vector<Link> links;

    //create links, fully connected
    //for each neuron of a layer, connect them to all neurons of the next layer
    //if not bias
    for(unsigned int i = 0; i < m_layersCount-1; i++)//for each layer (except output)
    {
		
		for(unsigned int j = 0; j < layers[i].neurons.size(); j++)//for each neurons
        {
			
			for(unsigned int k = 0; k < layers[i+1].neurons.size(); k++)//for each neuron of the next layer
            {
				              
				if(m_layers[i+1].neurons[k].type != NType::BIAS)
                {
                    Link l;
                    l.a = m_layers[i].neurons[j].id;
                    l.b = m_layers[i+1].neurons[k].id;
                    links.push_back(l);
                }
            }
        }
    }

	m_neuronsCount = neurons.size();
	m_linksCount = links.size();
	m_neurons = neurons;
	m_links = links;
}

FFNet::~FFNet()
{

}

std::vector<Layer> FFNet::getLayers()
{
    return m_layers;
}

std::vector<Neuron> FFNet::layers2vector()
{
    std::vector<Neuron> neurons;
    unsigned int id = 0;

    //Store all neurons from layers to vector
    //for each layer, push all the neurons to vector
    for(unsigned int i = 0; i < m_layersCount; i++)//for each layer
    {

        for(unsigned int j = 0; j < m_layers[i].neurons.size(); j++)
        {
			
            m_layers[i].neurons[j].id = id;
			m_layers[i].neurons[j].parent = this;
            neurons.push_back(m_layers[i].neurons[j]);
            id++;
        }
    }

    return neurons;
}