using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.Weights.Factory;

namespace PerceptronPP;

internal static class Program
{
	public static void Main()
	{
		var network = new Network(new EmptyComputable(), 2,1,2);//3,4,3
		network
			.SetWeights(new ConstantWeightFactory(new[] { new[,] { { 0.5 }, { 2.0 } }, new[,] { { 1, 1.0 } } }))
			//.SetWeights(new RandomWeightsFactory(network.GetNeuronCount))
			.SetBiases(new BiasConstantProvider(0));

		double[] result;
		var expected = new double[] { 0,1};
		for (int i = 0; i < 100; i++)
        {
			result = network.Compute(1, 1);
			Console.WriteLine(network.CalculateCost(result,expected));
			Tools.PrintDoubleArray(result);
			Console.WriteLine();
			network.BackPropagate(result, expected);
			network.GradientDescend(0.01);
		}
		//var result = network.Compute(1,1);
		//Tools.PrintDoubleArray(result);

		//network.BackPropagate(result, new[] { 3.0, 3.0, 3.0 });
		//network.BackPropagate(result, new[] { 0,1d});





		//var network = new Network(new EmptyComputable(), 3, 3);//3,4,3
		//network
		//	.SetWeights(new ConstantWeightFactory(new[] { new[,] { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1d } } }))
		//	//.SetWeights(new RandomWeightsFactory(network.GetNeuronCount))
		//	.SetBiases(new BiasConstantProvider(0));
		//var result = network.Compute(1, 1, 1);
		//Tools.PrintDoubleArray(result);

		////network.BackPropagate(result, new[] { 3.0, 3.0, 3.0 });
		//network.BackPropagate(result, new[] { 2.5, 2.5, 3 });

	}
}

//network.Layers[0].Weights = DenseMatrix.OfArray(new double[,] { { 0,0,0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } });