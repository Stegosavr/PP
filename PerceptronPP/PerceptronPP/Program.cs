using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.Providers;

namespace PerceptronPP;

internal static class Program
{
	public static void Main()
	{
		var network = new Network(new SigmoidComputable(), 3, 3);
		network
			.SetWeights(new RandomWeightsProvider(network.GetNeuronCount))
			.SetBiases(new BiasConstantProvider(1));
		var result = network.Compute(1, 1, 1);
		PrintDoubleArray(result);
	}

	private static void PrintDoubleArray(double[] array)
	{
		if (array.Length == 0) return;
		Console.Write(array[0]);
		foreach (var number in array.Skip(1))
			Console.Write($", {number}");
		Console.WriteLine();
	}
}

//network.Layers[0].Weights = DenseMatrix.OfArray(new double[,] { { 0,0,0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } });