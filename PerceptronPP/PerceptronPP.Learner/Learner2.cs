using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.GradientDescent.Optimizers;
using PerceptronPP.Core.FileManager.Excel;
using System.Diagnostics;

namespace PerceptronPP.Learner;

public class Learner2
{
	private readonly Network _network;
	private readonly IOptimizer _optimizer;
	public Learner2(Network network, IOptimizer optimizer)
	{
		_network = network;
		_optimizer = optimizer;
	}

	public void LoopLearn
	(int batchSize, double learningCoefficient,
	double[][] trainData, int[] expectedOutputs,
	double[][] testData, int[] expectedTestOutputs,
	Parameters parameterToChange, double changeTo,
	int iterationsNumber)
	{
		double learningDelta = 0;
		int batchDelta = 0;
		double result = 0;

		switch (parameterToChange)
        {
			case (Parameters.LearningCoefficient):
				learningDelta = (changeTo - learningCoefficient) / iterationsNumber;
				break;
			case (Parameters.BatchSize):
				batchDelta = (int)((changeTo - batchSize - 1) / iterationsNumber);
				break;
		}
		for (int iteration = 0; iteration < iterationsNumber; iteration++)
		{
			Stopwatch stopWatch = new Stopwatch();
			stopWatch.Start();

			var network = Learn(_network, trainData, expectedOutputs, batchSize, learningCoefficient);
			
			stopWatch.Stop();
			TimeSpan ts = stopWatch.Elapsed;

			result = Test(network, testData, expectedTestOutputs);
			ExcelWriter.SaveNetworkParameters(
				_network, _optimizer, batchSize,
				parameterToChange is Parameters.LearningCoefficient ? (int)changeTo : 0,
				learningCoefficient,
				parameterToChange is Parameters.BatchSize ? changeTo : 0,
				trainData.Length, result, ts.TotalSeconds);
			learningCoefficient += learningDelta;
			batchSize += batchDelta;
		}
	}

	public Network Learn(Network initialNetwork, IEnumerable<double[]> trainingData, 
		IEnumerable<int> expectedOutputs, 
		int batchSize, double learningCoefficient, 
		int toBatchSize = 0, double toLearningCoefficient = 0)
	{
		var network = initialNetwork.Clone();
		var optimizer = _optimizer.Clone();

		using var trainingEnumerator = trainingData.GetEnumerator();
		using var labelsEnumerator = expectedOutputs.GetEnumerator();
		var i = 0;
		var lastCost = 0d;
		for (;; i++)
		{
			var isTrainingMoveCompleted = trainingEnumerator.MoveNext();
			var isLabelsMoveCompleted = labelsEnumerator.MoveNext();
			if (isTrainingMoveCompleted != isLabelsMoveCompleted)
				throw new Exception("Mismatch Images and Names count!");
			if (!isTrainingMoveCompleted) break;

			var expectedOutput = IntToExpectedDoubleMatrix(labelsEnumerator.Current);
			var output = network.ComputeMatrix(trainingEnumerator.Current);
			network.BackPropagate(output, expectedOutput);
			network.CalculateCost(output, expectedOutput);
			if ((i + 1) % batchSize != 0) continue;

			network.GradientDescent(optimizer, learningCoefficient);
			lastCost = network.GetCost() / batchSize;
			if ((i + 1) % 100 == 0)
				WriteStepData(lastCost, i);
			network.ResetCost();
		}

		if ((i) % batchSize == 0) return network;
		network.GradientDescent(_optimizer, learningCoefficient);
		var correctedLastCost = network.GetCost() / (i % batchSize);
		WriteStepData(correctedLastCost, i);

		var (_, top) = Console.GetCursorPosition();
		Console.SetCursorPosition(0, top + 2);

		return network;
	}

	private static void WriteStepData(double cost, int i)
	{
		Console.WriteLine(cost);
		Console.WriteLine(i);
		var (_, top) = Console.GetCursorPosition();
		Console.SetCursorPosition(0, top - 2);
	}



	private static double[] IntToExpectedDoubleArray(int number)
	{
		var data = new double[10];
		data[number] = 1;
		return data;
	}

	private static Matrix<double> IntToExpectedDoubleMatrix(int number)
	{
		return Layer.MatrixArray(IntToExpectedDoubleArray(number));
	}

	public double Test(Network network, double[][] testData, int[] expectedOutputs)
	{
		int count = 0;
		int correct = 0;
		double[] output;
		for (int i = 0; i < testData.Length; i++)
		{
			output = Iterate(network, testData[i], expectedOutputs[i]);
			if (Check(output, expectedOutputs[i]))
				correct++;
			count++;

			if ((i + 1) % 100 == 0)
				WriteStepData(correct / (double)count * 100, i);

		}
		var (_, top) = Console.GetCursorPosition();
		Console.SetCursorPosition(0, top+5);

		return correct / (double)count * 100;
	}

	public static double[] Iterate(Network network, double[] input, int expectedOutput, bool backProp = false)
	{
		var output = network.Compute(input);
		if (backProp)
			network.BackPropagate(output, Learning.IntToExpectedOutputArray(expectedOutput));
		network.CalculateCost(output, Learning.IntToExpectedOutputArray(expectedOutput));

		return output;
	}

	private static bool Check(double[] output, int expectedOutput)
	{
		var label = expectedOutput;
		var outputLabel = output.Select((e, i) => Tuple.Create(e, i)).OrderByDescending(tuple => tuple.Item1).First().Item2;
		if (label == outputLabel) return true;
		return false;
	}
}