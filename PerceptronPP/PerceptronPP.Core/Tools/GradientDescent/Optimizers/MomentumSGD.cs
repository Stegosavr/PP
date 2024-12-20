﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using PerceptronPP.Core.Tools.GradientDescent;

namespace PerceptronPP.Core.Tools.GradientDescent.Optimizers;

public class MomentumSGD : IOptimizer
{
    public string Name { get; } = nameof(MomentumSGD);
    private List<GradientDescentData> _gradientDescentData;
    private readonly double _beta;

    public MomentumSGD(double beta,int layersCount)
    {
        //_gradientDescentData = Enumerable.Repeat(new GradientDescentData(), layersCount-1).ToList();
        _gradientDescentData = new List<GradientDescentData>();
        for (int i = 0; i < layersCount - 1; i++)
            _gradientDescentData.Add(new GradientDescentData());
        _beta = beta;
    }

    public void GradientDescent(ParameterType type, ref Matrix<double> currentParameters, 
        Matrix<double> parametersDerivative, double coefficient, int layerIndex)
    {
        if (_gradientDescentData[layerIndex].GetPreviousVelocity(type) == null)
        {
            _gradientDescentData[layerIndex].SetPreviousVelocity(type, parametersDerivative);
            currentParameters -= parametersDerivative * coefficient;
        }
        else
        {
            var velocity = _gradientDescentData[layerIndex].GetPreviousVelocity(type)
                * _beta + (1 - _beta) * parametersDerivative;
            _gradientDescentData[layerIndex].SetPreviousVelocity(type, velocity);
            currentParameters -= velocity * coefficient;
        }
    }

    public IOptimizer Clone()
    {
        return new MomentumSGD(_beta, _gradientDescentData.Count + 1);
    }
}
