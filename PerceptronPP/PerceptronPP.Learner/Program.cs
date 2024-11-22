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
	private static readonly int[] NeuronCounts = new[] { 784, 30, 30, 10 };

    //public static void Test()
    //{
    //    var network = new Network(Computable, new[] {3,2,4});
    //    network.SetWeights(new ConstantWeightFactory(new[] {
    //        new double[,] { { 0, 0 }, { 0.5, 0.5}, { 0.5, 0.5 } },
    //        new double[,] { { 1.0 }, { 2.0 } } }))
    //        .SetBiases(new BiasConstantProvider(2));
    //    WeightsOperator.SaveToFile(network, "test.txt");
    //    var weights = WeightsOperator.ReadFromFile("test.txt");
    //    var newNetwork = new Network(Computable, weights.Item3).SetWeights(weights.Item1).SetBiases(weights.Item2);
    //}

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

		var learner = new Learner2(network, new StochasticGradientDescent(), "09-06-24.xlsx");
        //new RMSPropagation(0.9, network.GetLayerCount()
        //Console.WriteLine($"Learning elapsed {learnTime.ToString()}");

        using var testImages = new IdxReader(@"..\..\..\..\Datasets\MNIST\t10k-images.idx3-ubyte");
		using var testLabels = new IdxReader(@"..\..\..\..\Datasets\MNIST\t10k-labels.idx1-ubyte");

		var trainingMnist = Tuple.Create(GetSample(imagesData).ToArray(), GetLabel(labelsData).ToArray());
		var testMnist = Tuple.Create(GetSample(testImages).ToArray(), GetLabel(testLabels).ToArray());

		//learner.Test(learnTime, GetSample(testImages).ToArray(), GetLabel(testLabels).ToArray());
		//WeightsOperator.SaveToFile(network, "08-06-24.txt");
		//var weights = WeightsOperator.ReadFromFile("29-05-24.txt");
		//network.SetBiases(weights.Item2).SetWeights(weights.Item1);

        learner.LoopLearn(1, 3, trainingMnist.Item1, trainingMnist.Item2,
            testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
            1, 1);

        //var learnTime = learner.Learn(network, trainingMnist.Item1.ToArray(), 
        //	trainingMnist.Item2.ToArray(), 1, 3);
        //WeightsOperator.SaveToFile(learnTime, "08-06-24.txt");


        //for (var j = 0; j < 10; j++)
        //{
        //    for (var i = 0.5; i > 0.01; i -= 0.1)
        //    {
        //        network.SetWeights(new RandomWeightsFactory(network.GetNeuronCount, WeightsProviderType.GaussianRandom, i));
        //        learner = new Learner2(network, new StochasticGradientDescent(), "03-06-24(5).xlsx");
        //        learner.LoopLearn(1, 3, trainingMnist.Item1, trainingMnist.Item2,
        //            testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //            1, 1, i);
        //    }
        //}
        //for (var i = 0.3; i > 0; i -= 0.05)
        //{
        //    learner.LoopLearn(5, i, trainingMnist.Item1, trainingMnist.Item2,
        //        testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //        5, 1);
        //}


        //learner.LoopLearn(30, 0.01, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 30);
        //learner.LoopLearn(30, 0.05, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 30);
        //learner.LoopLearn(20, 0.1, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 20);
        //learner.LoopLearn(30, 0.2, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 30);
        //learner.LoopLearn(20, 0.3, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 20);
        //learner.LoopLearn(30, 0.4, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 30);
        //learner.LoopLearn(20, 0.5, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 20);
        //learner.LoopLearn(20, 0.75, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 20);
        //learner.LoopLearn(20, 1, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 20);
        //learner.LoopLearn(20, 1.5, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 20);
        //learner.LoopLearn(20, 2, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 20);
        //learner.LoopLearn(30, 3, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 20);
        //learner.LoopLearn(30, 5, trainingMnist.Item1, trainingMnist.Item2,
        //    testMnist.Item1, testMnist.Item2, Parameters.BatchSize,
        //    1, 20);

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