using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace PerceptronPP.Core.Tools.Optimizers
{
    public class StochasticGradientDescent : IOptimizer
    {
        public string Name { get; } = nameof(StochasticGradientDescent);

        public double GradientDescent(Matrix<double> currentParameters, 
            GradientDescentData data, double learningCoefficient)
        {
            throw new NotImplementedException();
        }
    }
}
