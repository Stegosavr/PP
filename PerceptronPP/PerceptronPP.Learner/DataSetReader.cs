using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.IO;

namespace PerceptronPP.Learner;

public static class MnistDataSetReader
{
	public static Tuple<IEnumerable<double[]>, IEnumerable<int>> LoadMnistTrain()
	{
		using var images = new IdxReader(@"..\..\..\..\Datasets\MNIST\train-images.idx3-ubyte");
		using var labels = new IdxReader(@"..\..\..\..\Datasets\MNIST\train-labels.idx1-ubyte");
		return Tuple.Create(GetSample(images), GetLabel(labels));
	}

	public static Tuple<IEnumerable<double[]>, IEnumerable<int>> LoadMnistTest()
	{
		using var images = new IdxReader(@"..\..\..\..\Datasets\MNIST\t10k-images.idx3-ubyte");
		using var labels = new IdxReader(@"..\..\..\..\Datasets\MNIST\t10k-labels.idx1-ubyte");
		return Tuple.Create(GetSample(images), GetLabel(labels));
	}

	public static IEnumerable<double[]> GetSample(IdxReader reader)
		{
		for (var i = 0; i < reader.Samples; i++)
		{
			yield return ((byte[])reader.ReadVector()).Select(e => (int)e / 255.0).ToArray();
		}
	}

	public static IEnumerable<int> GetLabel(IdxReader reader)
	{
		for (var i = 0; i < reader.Samples; i++)
		{
			yield return (byte)reader.ReadValue();
		}
	}
}
