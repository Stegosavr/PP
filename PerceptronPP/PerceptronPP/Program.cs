using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.Weights.Factory;

namespace PerceptronPP;

internal static class Program
{
	public static void Main()
	{
		var network = new Network(new SigmoidComputable(), 3, 3);
		network
			.SetWeights(new RandomWeightsFactory(network.GetNeuronCount))
			.SetBiases(new BiasConstantProvider(1));
		var result = network.Compute(1, 1, 1);
		Tools.PrintDoubleArray(result);
	}
}

//network.Layers[0].Weights = DenseMatrix.OfArray(new double[,] { { 0,0,0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } });