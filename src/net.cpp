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

#include "../include/net.hpp"
#include <random>
#include <iostream>

using namespace snn;
/*****************************Neuron**************************/
float Neuron::compute()
{
    switch(type)
    {
        case NType::INPUT:
            output = parent->m_inputs[id];
            break;
        case NType::HIDDEN:
            output = parent->m_hidFun(output);
            break;
        case NType::BIAS:
            //output already defined
            break;
		/*
        case NType::OUTPUT:
            output = parent->m_outFun(output);
            break;*/
    }
    
	return output;
}

/*******************************Net***************************/

Net::Net(std::vector<Neuron> neurons, std::vector<Link> links)
{
	std::cout << neurons.size() << "\t" << links.size();
    m_neurons = neurons;
    m_neuronsCount = neurons.size();

    m_links = links;
	m_linksCount = links.size();

    for(unsigned int i = 0; i < m_neuronsCount-1; i++)
    {
		m_neurons[i].id = i;
        m_neurons[i].parent = this;
    }
}

Net::~Net()
{

}

std::vector<float> Net::feed(std::vector<float> inputs)
{
    m_inputs = inputs;

    unsigned int actualID = 1; //the first neuron can't have 1 as id


    //m_links: feedforward links, and stored by order
    //When in the list the previous neuron is not the same as the actual
    //we compute it, because we know that there are no dependencies in the next neurons
    for(unsigned int i = 0; i < m_linksCount; i++)
    {
		unsigned int idA = m_links[i].a;
		unsigned int idB = m_links[i].b;
		float weight = m_links[i].weight;
        if(actualID != m_neurons[idA].id)//if actualneuron != link.a
        {
			m_neurons[idB].output += m_neurons[idA].compute()*weight;
        }
        else //if actualneuron == link.a
        {
            m_neurons[idB].output += m_neurons[idA].output*weight;//no need to recompute
        }
        actualID = m_neurons[idA].id;
    }

    //find all outputs neurons, compute them, return them w/ outputs vector
    std::vector<float> outputs;
    for(unsigned int i = 0; i < m_neuronsCount; i++)
    {
        if(m_neurons[i].type == NType::OUTPUT)
        {
            outputs.push_back(m_outFun(m_neurons[i].output));
        }
    }

    return outputs;

}

void Net::setOutputFunction(std::function<float(float)> actFun)
{
    m_outFun = actFun;
}

void Net::setHiddenFunction(std::function<float(float)> actFun)
{
    m_hidFun = actFun;
}

std::function<float(float)> Net::getOutputFunction()
{
    return m_outFun;
}
std::function<float(float)> Net::getHiddenFunction()
{
    return m_hidFun;
}


unsigned int Net::getSize()
{
    return m_neurons.size();
}

void Net::randWeights()
{
    //TODO: randWeights()
}

void Net::setLinks(std::vector<Link> links)
{
    m_links = links;
}

std::vector<Link> Net::getLinks()
{
    return m_links;
}