using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.GradientDescent.Optimizers;

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

	public double[] Learn(IEnumerable<double[]> trainingData, IEnumerable<int> expectedOutputs, uint batchSize,
		double learningCoefficient)
	{
		var network = new Network(_computable, _neuronCounts);
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
			BackPropagate(network, output, expectedOutput);
			CalculateCost(network, output, expectedOutput);
			if ((i + 1) % batchSize != 0) continue;

			network.GradientDescent(new RMSPropagation(0.9, network._neuronCounts.Length), learningCoefficient);
			lastCost = network.GetCost() / batchSize;
			WriteStepData(lastCost, i);
			network.ResetCost();
		}

		if ((i) % batchSize == 0) return new[] { lastCost, i };
		network.GradientDescent(new RMSPropagation(0.9, network._neuronCounts.Length), learningCoefficient);
		var correctedLastCost = network.GetCost() / (i % batchSize);
		WriteStepData(correctedLastCost, i);
		return new[] { correctedLastCost, i };
	}

	private static void WriteStepData(double cost, int i)
	{
		Console.WriteLine(cost);
		Console.WriteLine(i);
		var (_, top) = Console.GetCursorPosition();
		Console.SetCursorPosition(0, top - 2);
	}

	public static double CalculateCost(Network network, Matrix<double> input, Matrix<double> expectedOutput)
	{
		return 0;
	}

	public void BackPropagate(Network network, Matrix<double> output, Matrix<double> expectedOutput)
	{
		var data = 2 * output - expectedOutput;
		// Enumerable.Range(0, _neuronCounts.Length - 1).Select(i =>
		// 	network.GetLayer(i).BackPropagate()
		// 	);
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
}