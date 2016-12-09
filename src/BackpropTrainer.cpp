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

#include "../include/BackpropTrainer.hpp"
#include "../include/util.hpp"

#include <iostream>

using namespace snn;

BackpropTrainer::BackpropTrainer(unsigned int _backpropType, float _learningRate, float _momentum)
{
    backpropType_m = _backpropType;
    learningRate_m = _learningRate;
    momentum_m = _momentum;
}

BackpropTrainer::~BackpropTrainer()
{

}

void BackpropTrainer::setOutputDerivative(std::function<float(float)> _actFunDerivative)
{
    outputDer_m = _actFunDerivative;
}

void BackpropTrainer::setHiddenDerivative(std::function<float(float)> _actFunDerivative)
{
    hiddenDer_m = _actFunDerivative;
}

void BackpropTrainer::train(Net* _net, unsigned int _epochs, std::vector<Dataset> _datasets)
{
    _net->randWeights(-1.0f, 1.0f);

    if(backpropType_m == BackpropType::ONLINE)
        this->trainOnline(_net, _epochs, _datasets);
    else
        this->trainBatch(_net, _epochs, _datasets);
}

void BackpropTrainer::trainOnline(Net* _net, unsigned int _epochs, std::vector<Dataset> _datasets)
{

    //The backpropagation works as the following:
    //for each iteration defined on a criteria (here epochs)
    //and for each example from the dataset (inputs & expected results)
    //1) compute the Error of outputs: E = expected - observed
    //2) compute the delta of outputs: f'(Oi)*E, 
    //  (where f'(x) is the derivative of the used activation function to feed,
    //  and Oi the output from output neuron i)
    //3) compute the delta of hidden neurons: f'(Oi)*sum( dk*wik ))
    //  where dk means the delta from the neuron k connected to the neuron i
    //  the link is i -> k
    //  and wik means the weight of the link connected from i to k (as neurons)
    //4) then compute the weight update: dw(t) = -lr * di * Oi + a * dw(t-1)
    //  where lr is the learningRate
    //  di is the delta of neuron i
    //  Oi is the output of neuron i
    //  a is the momentum
    //  dw(t) means the actual weight delta that we're computing
    //  and dw(t-1) means the old weight update
    //5) then just compute w = w + dw

    // find the number of outputs

    for(unsigned int i = 0; i < _epochs; i++)
    {
        for(unsigned int j = 0; j < _datasets.size(); j++)
        {
            //step 0
            std::vector<float> outputs = _net->feed(_datasets[j].inputs); //feed the net, sets outputs,sums
            
            std::vector<Neuron> neurons = _net->getNeurons();
            //step 1 & 2
            unsigned int outputsCount = _datasets[j].outputs.size();
            for (unsigned int k = neurons.size() - outputsCount; k < neurons.size(); k++)
            {
                float output = neurons[k].output;
                float sum = neurons[k].sum;
                float result = (_datasets[j].outputs[ k - (neurons.size() - outputsCount) ] - output );
                neurons[k].delta = outputDer_m(output)*result;
                neurons[k].isDeltaComputed = true;
            }

            for(unsigned int k = 0; k < neurons.size()-outputsCount; k++)
            {

                neurons[k].isDeltaComputed = false;

            }

            std::vector<Link> links = _net->getLinks();
            
            //step 3
            float sum = 0.0f;
            for(unsigned int k = neurons.size()-1; k > 0; k--)
            {
                for(unsigned int l = links.size()-1; l > 0; l--)
                {
                    unsigned int idA = links[l].a;
                    unsigned int idB = links[l].b;

                    if( (idA == k) && ( neurons[idA].type != NType::BIAS) 
                        && neurons[idB].isDeltaComputed )
                    {
                        sum += neurons[idB].delta * links[l].weight;
                    }
                }
                if(k == neurons.size()-1)
                    continue;

                neurons[k].delta =  hiddenDer_m(neurons[k].output) * sum;
                //std::cout << "delta k:\t" << neurons[k].delta << "\n";
                neurons[k].isDeltaComputed = true;
                sum = 0.0f;
            }

            //step 4 & 5
            for(unsigned int k = 0; k < links.size(); k++)
            {
                Neuron b = neurons[ links[k].b ];
                Neuron a = neurons[ links[k].a ];
                float update = -learningRate_m * a.output * b.delta + momentum_m * links[k].lastUpdate;
                links[k].weight += update;
                links[k].lastUpdate = update;
            }

            _net->setNeurons(neurons); //to access deltas
            _net->setLinks(links); //to update previous & actual weight
        }
    }
}
void BackpropTrainer::trainBatch(Net* _net, unsigned int _epochs, std::vector<Dataset> _datasets)
{
	// TODO: trainBatch()
}


