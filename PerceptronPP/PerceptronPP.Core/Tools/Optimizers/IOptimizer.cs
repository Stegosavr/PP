using MathNet.Numerics.LinearAlgebra;
namespace PerceptronPP.Core.Tools.Optimizers;

public interface IOptimizer
{
	public string Name { get; }
	public double GradientDescent(Matrix<double> currentParameters, 
		GradientDescentData data, double learningCoefficient);
}