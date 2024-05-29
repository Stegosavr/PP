using Accord.IO;
using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.GradientDescent.Optimizers;
using PerceptronPP.Core.Tools.Weights.Factory;

namespace PerceptronPP.Learner;

public static class Program
{
	private static readonly IComputable Computable = new SigmoidComputable();
	private static readonly int[] NeuronCounts = new[] { 784, 16, 16, 10 };

	public static void Main()
	{
		Console.WriteLine("Neural Network Info:");
		Console.WriteLine($"\tLayers count: {NeuronCounts.Length}");
		Console.WriteLine($"\tNeuron counts:{NeuronCounts.Aggregate("", ((s, i) => $"{s} {i.ToString()}"))}");
		Console.WriteLine($"\tComputable Function: {Computable.Name}");
		var network = new Network(Computable, NeuronCounts);
		network
			.SetWeights(new RandomWeightsFactory(network.GetNeuronCount, WeightsProviderType.Random, 0.5))
			.SetBiases(new BiasConstantProvider(0));

		using var imagesData = new IdxReader(@"../../../../Datasets/MNIST/train-images.idx3-ubyte");
		using var labelsData = new IdxReader(@"../../../../Datasets/MNIST/train-labels.idx1-ubyte");

		var learner = new Learner2(network, new RMSPropagation(0.9,network.GetLayerCount()));

		//Console.WriteLine($"Learning elapsed {learnTime.ToString()}");

		using var testImages = new IdxReader(@"..\..\..\..\Datasets\MNIST\t10k-images.idx3-ubyte");
		using var testLabels = new IdxReader(@"..\..\..\..\Datasets\MNIST\t10k-labels.idx1-ubyte");

		var trainingMnist = Tuple.Create(GetSample(imagesData).ToArray(), GetLabel(labelsData).ToArray());
		var testMnist = Tuple.Create(GetSample(testImages).ToArray(), GetLabel(testLabels).ToArray());

		//learner.Test(learnTime, GetSample(testImages).ToArray(), GetLabel(testLabels).ToArray());

		WeightsOperator.SaveToFile(network, "29-05-24.txt");

		var learnTime = learner.Learn(network, trainingMnist.Item1.Take(100).ToArray(), 
			trainingMnist.Item2.Take(100).ToArray(), 5, 0.05);


		learner.LoopLearn(30,0.01,trainingMnist.Item1,trainingMnist.Item2,
			testMnist.Item1,testMnist.Item2,Parameters.BatchSize,
			1,30);
		learner.LoopLearn(30, 0.05, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);
		learner.LoopLearn(30, 0.1, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);
		learner.LoopLearn(30, 0.2, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);
		learner.LoopLearn(30, 0.3, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);
		learner.LoopLearn(30, 0.4, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);
		learner.LoopLearn(30, 0.5, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);
		learner.LoopLearn(30, 0.75, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);
		learner.LoopLearn(30, 1, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);
		learner.LoopLearn(30, 1.5, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);
		learner.LoopLearn(30, 2, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);
		learner.LoopLearn(30, 3, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);
		learner.LoopLearn(30, 5, trainingMnist.Item1, trainingMnist.Item2,
			testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
			1, 30);

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