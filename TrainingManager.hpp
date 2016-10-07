#pragma once

#include <vector>
#include "ffnet.hpp"

//TODO: TrainingManager.cpp

namespace snn
{

    enum trainMethods: unsigned int
    {
        BACKPROP = 1,
        GENETIC
    };

    class Dataset
    {
        public:
            Dataset(std::vector<float> inputs, std::vector<float> outputs);
            ~Dataset();

        private:
            std::vector<float> m_inputs;
            std::vector<float> m_outputs;
    };

    class TrainingManager
    {
        public:
            TrainingManager(unsigned int method);
            ~TrainingManager();

            void train(snn::FFNet *net, std::vector<Dataset> ds);

        private:
            std::vector<Dataset> m_dss;
    };
}