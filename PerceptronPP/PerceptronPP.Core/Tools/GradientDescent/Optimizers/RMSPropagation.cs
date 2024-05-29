using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using PerceptronPP.Core.Tools.GradientDescent;

namespace PerceptronPP.Core.Tools.GradientDescent.Optimizers;

public class RMSPropagation : IOptimizer
{
    public string Name { get; } = nameof(StochasticGradientDescent);
    private List<GradientDescentData> _gradientDescentData;
    private readonly double _beta;

    public RMSPropagation(double beta,int layersCount)
    {
        //_gradientDescentData = Enumerable.Repeat(new GradientDescentData(), layersCount-1).ToList();
        _gradientDescentData = new List<GradientDescentData>();
        for (int i = 0; i < layersCount - 1; i++)
            _gradientDescentData.Add(new GradientDescentData());
        _beta = beta;
    }

    public IOptimizer Clone()
    {
        return new RMSPropagation(_beta, _gradientDescentData.Count + 1);
    }

    public void GradientDescent(ParameterType type, ref Matrix<double> currentParameters, 
        Matrix<double> parametersDerivative, double coefficient, int layerIndex)
    {
        Matrix<double> velocity;
        if (_gradientDescentData[layerIndex].GetPreviousVelocity(type) == null)
        {
            velocity = (1 - _beta) * parametersDerivative.PointwisePower(2);
            _gradientDescentData[layerIndex].SetPreviousVelocity(type, velocity);
        }
        else
        {
            velocity = _gradientDescentData[layerIndex].GetPreviousVelocity(type)
                * _beta + (1 - _beta) * parametersDerivative.PointwisePower(2);
            _gradientDescentData[layerIndex].SetPreviousVelocity(type, velocity);
        }
        currentParameters -= coefficient * parametersDerivative.PointwiseDivide((velocity.PointwiseSqrt()) + (1e-9) );
        
    }
}
