using System.Globalization;
using System.Text;
using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Weights.Factory;

namespace PerceptronPP;

public static class WeightsOperator
{
	public static Tuple<MapWeightsFactory, BiasMapProvider, int[]> ReadFromFile(string path)
	{
		var lines = File.ReadLines(Path.Combine("..", "..", "..", "..", "Weights", path), Encoding.UTF8);
		using var linesEnumerator = lines.GetEnumerator();
		if (!linesEnumerator.MoveNext()) throw new Exception("Empty file");
		var networkInfo = linesEnumerator.Current.Split(",").Select(x => int.Parse(x)).ToArray();
		if (networkInfo.Length == 0) throw new Exception("Wtf");

		var weights = new double[networkInfo.Length][,];
		var biases = new double[networkInfo.Length, networkInfo.Skip(1).Max()];
		var currentLine = 1;

		for (var i = 0; i < weights.Length - 1; i++)
		{
			weights[i] = new double[networkInfo[i], networkInfo[i + 1]];
			for (var x = 0; x < networkInfo[i]; x++)
			for (var y = 0; y < networkInfo[i + 1]; y++)
			{
				if (!linesEnumerator.MoveNext()) throw new Exception("File length mismatch with network info.");
				currentLine++;
				if (!double.TryParse(linesEnumerator.Current, out var n))
					throw new Exception($"Cannot parse number on {currentLine} lane.");
				weights[i][x, y] = n;
			}
		}

		var xLength = biases.GetLength(0);
		//var yLength = biases.GetLength(1);
		for (var x = 0; x < xLength-1; x++)
		{
			var yLength = networkInfo[x+1];
			for (var y = 0; y < yLength; y++)
			{
				if (!linesEnumerator.MoveNext()) throw new Exception("File length mismatch with network info.");
				currentLine++;
				if (!double.TryParse(linesEnumerator.Current, out var n))
					throw new Exception($"Cannot parse number on {currentLine} lane.");
				biases[x, y] = n;
			}
		}
		// foreach (var line in lines)
		// {
		// 	if (filling == false) { weights[currentLayer] = new double[layers[currentLayer],] }
		// }

		return new Tuple<MapWeightsFactory, BiasMapProvider, int[]>(
			new MapWeightsFactory(weights),
			new BiasMapProvider(biases),
			networkInfo
		);
	}

	public static void SaveToFile(IEnumerable<double[,]> weights, IEnumerable<double> biases, IEnumerable<int> layers,
		string path = "export.txt")
	{
		var info = string.Join(",", layers.Select(x => x.ToString()));
		var lines = Enumerable.Empty<string>().Append(info);
		lines = weights.SelectMany(w => w.Cast<double>()).Aggregate(lines,
			(current, l) => current.Append(l.ToString(CultureInfo.CurrentCulture)));
		lines = biases.Aggregate(lines, (current, b) => current.Append(b.ToString(CultureInfo.CurrentCulture)));
		var filepath = Path.Combine("..", "..", "..", "..", "Weights", path);
		using var outputFile = new StreamWriter(filepath);
		foreach (var line in lines)
			outputFile.WriteLine(line);
		Console.WriteLine($"Network exported` to {path}");
	}

	public static void SaveToFile(this Network network, string path = "export.txt")
	{
		var c = network.GetLayerCount();
		var weights = Enumerable.Empty<double[,]>();
		var biases = Enumerable.Empty<double>();
		weights = network.Aggregate(weights, (current, layer) => current.Append(layer.GetWeights().ToArray()));
		biases = network.SelectMany(layer => layer.GetBiases().ToArray().Cast<double>())
			.Aggregate(biases, (current, x) => current.Append(x));
		var layers = network.Select(x => x.GetNeuronsCount());
		SaveToFile(weights, biases, layers, path);
	}
}