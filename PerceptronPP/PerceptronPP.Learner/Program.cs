using Accord.IO;
using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.Weights.Factory;

namespace PerceptronPP.Learner;

internal static class Program
{
	private static readonly IComputable Computable = new SigmoidComputable();
	private static readonly int[] NeuronCounts = new[] { 784, 16, 16, 10 };

	public static void Main()
	{
		Console.WriteLine("Neural Network Info:");
		Console.WriteLine($"\tLayers count: {NeuronCounts.Length}");
		Console.WriteLine($"\tNeuron counts:{NeuronCounts.Aggregate("", ((s, i) => $"{s} {i.ToString()}"))}");
		Console.WriteLine($"\tComputable Function: {Computable.Name}");
		var network = new Network(Computable, NeuronCounts); //3,4,3
		network
			//.SetWeights(new ConstantWeightFactory(new[] { new[,] { { 0.5 }, { 2.0 } }, new[,] { { 1, 1.0 } } }))
			.SetWeights(new RandomWeightsFactory(network.GetNeuronCount, WeightsProviderType.Random, 1))
			.SetBiases(new BiasConstantProvider(0));

		using var imagesData = new IdxReader(@"../../../../Datasets/MNIST/train-images.idx3-ubyte");
		using var labelsData = new IdxReader(@"../../../../Datasets/MNIST/train-labels.idx1-ubyte");

		var learner = new Learner(Computable, NeuronCounts);

		var learnTime = learner.Learn(GetSample(imagesData), GetLabel(labelsData), 12, 1);
		Console.WriteLine($"Learning elapsed {learnTime.ToString()}");
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