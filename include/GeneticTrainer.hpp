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

#include "net.hpp"
namespace snn
{
    class GeneticTrainer
    {
        public:
            GeneticTrainer(unsigned int _populationPerGen, float _mutationRate, float _crossoverRate, unsigned int _eliteCopies);
            ~GeneticTrainer();

            void run(Net* _net, unsigned int _generations, std::function<float(Net)> _fitnessFunction);
            
        
        private:
            unsigned int populationPerGen_m;
            float mutationRate_m;
            float crossoverRate_m;
            unsigned int eliteCopies_m;

            std::vector<float> fitnesses_m;
            std::vector< std::vector<Link> > population_m;//Vector of Vector of Weights
            std::vector<Link> model_m;


			std::vector<Link> randomWeights();
	};
}
