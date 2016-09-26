#pragma once

#include <vector>
#include <functional>

namespace snn
{

    struct Neuron
    {
        float weight;

    };

    struct Layer
    {
        void setActivationFunction(std::function<float(float)> actFun);
        std::vector<float> feed(std::vector<float> inputs);

        unsigned int size();

        private:
            std::vector<Neuron> m_neurons;
            std::function<float(float)> m_actFun = snn::FUN_SIGMOID;
    };

    class FFNet
    {
        public:
            FFNet(unsigned int num_in, unsigned int num_hid, unsigned int num_out, unsigned int num_hidLayer);
            ~FFNet();

            std::vector<float> feed(std::vector<float> inputs);

        private:
            Layer m_inputs;
            Layer m_outputs;
            std::vector<Layer> m_hiddens;
            
    };

}

