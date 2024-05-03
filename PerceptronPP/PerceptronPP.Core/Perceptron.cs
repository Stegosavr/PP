using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace PerceptronPP.Core;

public class Perceptron
{
	// public readonly IComputable ActivationComputable;
	// int LayersCount;
	public Layer[] Layers;
	// int ForwardIterationsCount = 0;

	public Perceptron(IComputable activationComputable, params Layer[] layers)
	{
		// ActivationComputable = activationComputable;
		// LayersCount = layers.Length;
		Layers = layers;
		// foreach (var layer in layers)
		// 	layer.Network = this;
	}

	public void Randomize()
	{
		// for (var i = 0; i < LayersCount - 1; i++)
		// {
		// 	Layers[i].RandomizeWeights();
		// }
	}

	public void ForwardPropagation()
	{
		// for (var i = 0; i < LayersCount-1; i++)
		// {
		// 	Layers[i].ComputeOutput(Layers[i+1]);
		// }
	}

	public Matrix<double> GetOutput()
	{
		return DenseMatrix.OfArray(new double[,] { { } });
	}
}