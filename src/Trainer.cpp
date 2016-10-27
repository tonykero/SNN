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

#include "../include/Trainer.hpp"

using namespace snn;

/***************************Dataset***************************/
Dataset::Dataset(std::vector<float> in, std::vector<float> out)
{
    inputs = in;
    outputs = out;
}

/***************************Trainer***************************/
Trainer::Trainer(Net *net)
{
    m_pnet = net;
}

Trainer::~Trainer()
{
    delete m_pnet;
}