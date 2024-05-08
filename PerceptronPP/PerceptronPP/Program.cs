using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.Weights.Factory;

namespace PerceptronPP;

internal static class Program
{
	public static void Main()
	{
		var network = new Network(new EmptyComputable(), 3,3);//3,4,3
		network
			.SetWeights(new ConstantWeightFactory(new[] { new[,] { { 1,1,1 }, { 1,1,1 }, { 1,1,1d} } }))
			//.SetWeights(new RandomWeightsFactory(network.GetNeuronCount))
			.SetBiases(new BiasConstantProvider(0));
		var result = network.Compute(1, 1, 1);
		Tools.PrintDoubleArray(result);

		//network.BackPropagate(result, new[] { 3.0, 3.0, 3.0 });
		network.BackPropagate(result, new[] { 2.5, 2.5, 2.5 });

	}
}

//network.Layers[0].Weights = DenseMatrix.OfArray(new double[,] { { 0,0,0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } });