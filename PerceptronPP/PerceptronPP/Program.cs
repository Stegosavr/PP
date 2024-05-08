using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.Weights.Factory;

namespace PerceptronPP;

internal static class Program
{
	public static void Main()
	{
		var network = new Network(new AtanComputable(), 3,4, 6,10,340,3);//3,4,3
		network
			//.SetWeights(new ConstantWeightFactory(new[] { new[,] { { 0, 0, 0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } } }))
			.SetWeights(new RandomWeightsFactory(network.GetNeuronCount))
			.SetBiases(new BiasConstantProvider(1));
		var result = network.Compute(1, 1, 1);
		Tools.PrintDoubleArray(result);

		network.BackPropagate(result, new[] { 1.0, 1.0, 1.0 });
	}
}

//network.Layers[0].Weights = DenseMatrix.OfArray(new double[,] { { 0,0,0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } });