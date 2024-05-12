using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.Weights.Factory;
using Accord.IO;

namespace PerceptronPP;

internal static class Program
{
	public static void Main()
	{
		//using (var rider = new IdxReader(@"..\..\..\..\..\Datasets\MNIST\t10k-labels.idx1-ubyte")) { var f = rider.ReadVector(); }

		var network = new Network(new SigmoidComputable(), 784, 16, 16, 10); //3,4,3
		network
			//.SetWeights(new ConstantWeightFactory(new[] { new[,] { { 0.5 }, { 2.0 } }, new[,] { { 1, 1.0 } } }))
			.SetWeights(new RandomWeightsFactory(network.GetNeuronCount))
			.SetBiases(new BiasConstantProvider(0));

		using var mnist = new IdxReader(@"../../../../Datasets/MNIST/train-images.idx3-ubyte");
		using var mnist2 = new IdxReader(@"../../../../Datasets/MNIST/train-labels.idx1-ubyte");
		Learning.Learn(network,
			12,
			1,
			GetSample(mnist).ToArray(),
			GetLabel(mnist2).Select(Learning.IntToExpectedOutputArray).ToArray()
		);


		//double[] result;
		//var expected = new double[] { 0,1};
		//for (int i = 0; i < 100; i++)
		//      {
		//	Console.Clear();
		//	result = network.Compute(1, 1);
		//	network.CalculateCost(result, expected);
		//	Console.WriteLine("Cost: " + network.GetCost());
		//	network.ResetCost();
		//	Console.WriteLine("Output:");
		//	Tools.PrintDoubleArray(result);
		//	Console.WriteLine("Expected ouput: \n" + String.Join(", ", expected.Select(e => e.ToString())));
		//	Thread.Sleep(200);
		//	network.BackPropagate(result, expected);
		//	network.GradientDescend(0.01);
		//}
	}

	private static IEnumerable<int> GetLabel(IdxReader reader)
	{
		for (var i = 0; i < reader.Samples; i++)
		{
			yield return (byte)reader.ReadValue();
		}
	}

	private static IEnumerable<double[]> GetSample(IdxReader reader)
	{
		for (var i = 0; i < reader.Samples; i++)
		{
			yield return ((byte[])reader.ReadVector()).Select(e => e / 255.0).ToArray();
		}
	}
}

//network.Layers[0].Weights = DenseMatrix.OfArray(new double[,] { { 0,0,0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } });