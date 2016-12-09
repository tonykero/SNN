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

using namespace snn;

FFNet::FFNet(std::vector<Layer> _layers)
{
    #ifdef DEBUG
    //checks if the network has at least 2 Layers
    assert( _layers.size() > 1 );
    //at least an input & an output layer
    assert( _layers[0].type == NType::INPUT && _layers.back().type == NType::OUTPUT );
    #endif

    layersCount_m = _layers.size();
    layers_m = _layers;

    std::vector<Neuron> neurons = layers2neurons();

    std::vector<Link> links = link();

    neuronsCount_m = neurons.size();
    linksCount_m = links.size();
    neurons_m = neurons;
    links_m = links;
}

FFNet::~FFNet()
{

}

std::vector<Neuron> FFNet::layers2neurons()
{
    std::vector<Neuron> neurons;

    unsigned int id = 0;

    //Store all neurons from layers to vector
    //for each layer, push all the neurons to vector
    for (unsigned int i = 0; i < layers_m.size(); i++)//for each layer
    {
        for (unsigned int j = 0; j < layers_m[i].neurons.size(); j++)
        {

            layers_m[i].neurons[j].id = id;
            layers_m[i].neurons[j].parent = this;
            neurons.push_back(layers_m[i].neurons[j]);
            id++;
        }
    }

    return neurons;
}

std::vector<Link> FFNet::link()
{

    std::vector<Link> links;

    //create links, fully connected
    //for each neuron of a layer, connect them to all neurons of the next layer
    //if not bias
    for (unsigned int i = 0; i < layers_m.size() - 1; i++)//for each layer (except output)
    {

        for (unsigned int j = 0; j < layers_m[i].neurons.size(); j++)//for each neurons
        {
            for (unsigned int k = 0; k < layers_m[i + 1].neurons.size(); k++)//for each neuron of the next layer
            {

                if (layers_m[i + 1].neurons[k].type != NType::BIAS)
                {
                    Link l;
                    l.a = layers_m[i].neurons[j].id;
                    l.b = layers_m[i + 1].neurons[k].id;
                    links.push_back(l);
                }
            }
        }
    }

    return links;
}
