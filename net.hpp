#pragma once

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

    struct Link
    {
        Neuron* a;
        Neuron* b;

        float weight = 1.0f;
    };

    struct Neuron
    {
        Net* parent;
        float output = 0.0f;
        unsigned int type;
        unsigned int id;
        
        void compute();

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

            std::vector<float> m_inputs;

            std::function<float(float)> m_outFun = snn::FUN_SIGMOID;
            std::function<float(float)> m_hidFun = snn::FUN_SIGMOID;

            friend struct Neuron;
    };

}

