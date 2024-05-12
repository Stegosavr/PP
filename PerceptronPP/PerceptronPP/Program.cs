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

			var network = new Network(new SigmoidComputable(), 784, 16, 16,10);//3,4,3
		network
			//.SetWeights(new ConstantWeightFactory(new[] { new[,] { { 0.5 }, { 2.0 } }, new[,] { { 1, 1.0 } } }))
			.SetWeights(new RandomWeightsFactory(network.GetNeuronCount))
			.SetBiases(new BiasConstantProvider(0));

        IEnumerable<double[]> GetSample(IdxReader reader)
		{
			for (int i=0;i<reader.Samples;i++)
			{
				yield return ((byte[])reader.ReadVector()).Select(e=>(int)e/255.0).ToArray();
			}
		}
        IEnumerable<int> GetLabel(IdxReader reader)
        {
            for (int i = 0; i < reader.Samples; i++)
            {
                yield return (byte)reader.ReadValue();
            }
        }


        using var mnist = new IdxReader(@"..\..\..\..\Datasets\MNIST\train-images.idx3-ubyte");
        using var mnist2 = new IdxReader(@"..\..\..\..\Datasets\MNIST\train-labels.idx1-ubyte");


        var images = GetSample(mnist).ToArray();
        var labels = GetLabel(mnist2).ToArray();
        var labeledImages = labels.Select((label,i) => Tuple.Create(label,images[i]));
        var sorted = labeledImages.OrderBy(x => x.Item1).Select(tuple => Tuple.Create(Learning.IntToExpectedOutputArray(tuple.Item1),tuple.Item2));
        var input1 = sorted.Select(x => x.Item1).ToArray();
        var input2 = sorted.Select(x => x.Item2).ToArray();



        Learning.Learn(network, 12, 1, input2, input1);










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
}

//network.Layers[0].Weights = DenseMatrix.OfArray(new double[,] { { 0,0,0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } });