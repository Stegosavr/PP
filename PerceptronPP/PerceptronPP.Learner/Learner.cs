using System.Diagnostics;
using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Computable;

namespace PerceptronPP.Learner;

public class Learner
{
	private readonly IComputable _computable;
	private readonly int[] _neuronCounts;

	public Learner(IComputable computable, params int[] neuronCounts)
	{
		_computable = computable;
		_neuronCounts = neuronCounts;
	}

	public TimeSpan Learn(IEnumerable<double[]> trainingData, IEnumerable<int> expectedOutputs, uint batchSize,
		double learningCoefficient)
	{
		var stopwatch = new Stopwatch();
		stopwatch.Reset();
		stopwatch.Start();
		Console.WriteLine("\n");
		var network = new Network(_computable, _neuronCounts);
		using var trainingEnumerator = trainingData.GetEnumerator();
		using var labelsEnumerator = expectedOutputs.GetEnumerator();
		var i = 0;
		for (;; i++)
		{
			var isTrainingMoveCompleted = trainingEnumerator.MoveNext();
			var isLabelsMoveCompleted = labelsEnumerator.MoveNext();
			if (isTrainingMoveCompleted != isLabelsMoveCompleted)
				throw new Exception("Mismatch Images and Names count!");
			if (!isTrainingMoveCompleted) break;

			var expectedOutput = IntToExpectedDoubleArray(labelsEnumerator.Current);
			var output = network.Compute(trainingEnumerator.Current);
			network.BackPropagate(output, expectedOutput);
			network.CalculateCost(output, expectedOutput);
			if ((i + 1) % batchSize != 0) continue;
			
			network.GradientDescend(learningCoefficient);
			var (_, top) = Console.GetCursorPosition();
			Console.SetCursorPosition(0, top - 2);
			Console.WriteLine(network.GetCost() / batchSize);
			Console.WriteLine(i);
			network.ResetCost();
		}
		stopwatch.Stop();

		if ((i) % batchSize == 0) return stopwatch.Elapsed;
		network.GradientDescend(learningCoefficient);
		var (_, lastTop) = Console.GetCursorPosition();
		Console.SetCursorPosition(0, lastTop + 2);
		Console.WriteLine(network.GetCost() / (i % batchSize));
		Console.WriteLine(i);
		return stopwatch.Elapsed;
	}
	
	private static double[] IntToExpectedDoubleArray(int number)
	{
		var data = new double[10];
		data[number] = 1;
		return data;
	}
}