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
			//.SetWeights(new ConstantWeightFactory(new[] { new[,] { { 0.5 }, { 2.0 } }, new[,] { { 1, 1.0 } } }))
			.SetWeights(new RandomWeightsFactory(network.GetNeuronCount))
			.SetBiases(new BiasConstantProvider(0));

		double[] result;
		var expected = new double[] { 0,1};
		for (int i = 0; i < 100; i++)
        {
			Console.Clear();

			result = network.Compute(1, 1);
			network.CalculateCost(result, expected);
			Console.WriteLine("Cost: " + network.GetCost());
			network.ResetCost();
			Console.WriteLine("Output:");
			Tools.PrintDoubleArray(result);
			Console.WriteLine("Expected ouput: \n" + String.Join(", ", expected.Select(e => e.ToString())));

			Thread.Sleep(200);

			network.BackPropagate(result, expected);
			network.GradientDescend(0.01);
		}

		//Tools.PrintDoubleArray(result);
	}
}

//network.Layers[0].Weights = DenseMatrix.OfArray(new double[,] { { 0,0,0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } });