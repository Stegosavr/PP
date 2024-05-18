using PerceptronPP.Core;
using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.Weights.Factory;

using Accord.IO;
using PerceptronPP.Core.Tools.GradientDescent.Optimizers;

namespace PerceptronPP;

internal static class Program
{
    private static void Main2()
    {
        var network = new Network(new AtanComputable(), 3, 3);//3,4,3
        network
            //.SetWeights(new ConstantWeightFactory(new[] { new[,] { { 0.5 }, { 2.0 } }, new[,] { { 1, 1.0 } } }))
            .SetWeights(new ConstantWeightFactory(new[] { new[,] { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1d } } }))
            .SetBiases(new BiasConstantProvider(0));

        var output = network.Compute(new[] { 1, 1, 1d });
        network.CalculateCost(output, new[] { 1, 1, 1d });
        network.BackPropagate(output, new[] { 1, 1, 1d });
        network.GradientDescent(new StochasticGradientDescent(),0.05);
    }

    public static void Main()
    {
        //Main2();
        //return;

        var network = new Network(new AtanComputable(), 784, 30, 30,10);//3,4,3
		network
			//.SetWeights(new ConstantWeightFactory(new[] { new[,] { { 0.5 }, { 2.0 } }, new[,] { { 1, 1.0 } } }))
			.SetWeights(new RandomWeightsFactory(network.GetNeuronCount,WeightsProviderType.GaussianRandom,1))
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
        var dict = labeledImages.ToLookup(e => e.Item1, e => e.Item2);
        var Dict = dict.ToDictionary(e=>e.Key,e=>e.Skip(1).ToArray());
        var input1 = sorted.Select(x => x.Item1).ToArray();
        var input2 = sorted.Select(x => x.Item2).ToArray();

        int n;

        Learning.Learn(network, new MomentumSGD(0.9,network._neuronCounts.Length), 4, 2, images, labels);
        //for (n = 0; n < 1; n++) LERN(network, new[] { Dict[0].Take(6000).ToArray() }, new[] { 0 });
        //for (n = 0; n < 1; n++) LERN(network, new[] { Dict[0].Take(6000).ToArray(), (Dict[1].Take(5000)).ToArray() }, new[] { 0,1 });
        //for (n = 0; n < 1; n++) LERN(network, new[] { Dict[0].Take(5000).ToArray(), (Dict[1].Take(5000)).ToArray(), (Dict[2].Take(5000)).ToArray() }, new[] { 0,1,2 });
        //for (n = 0; n < 1; n++) LERN(network, new[] { Dict[0].Take(5000).ToArray(), (Dict[1].Take(5000)).ToArray(), (Dict[2].Take(5000)).ToArray(), (Dict[3].Take(5000)).ToArray() }, new[] { 0,1,2,3 });
        //for (n = 0; n < 1; n++) LERN(network, new[] { Dict[0].Take(5000).ToArray(), (Dict[1].Take(5000)).ToArray(), (Dict[2].Take(5000)).ToArray(), (Dict[3].Take(5000)).ToArray(), (Dict[4].Take(5000)).ToArray() }, new[] { 0,1,2,3,4 });
        //for (n = 0; n < 4; n++) LERN(network, new[] { Dict[0].Take(2500).ToArray(), (Dict[1].Take(2500)).ToArray(), (Dict[2].Take(2500)).ToArray(), (Dict[3].Take(2500)).ToArray(), (Dict[4].Take(2500)).ToArray(), (Dict[5].Take(2500)).ToArray() }, new[] { 0, 1, 2, 3, 4 ,5});
        //for (n = 0; n < 5; n++) LERN(network, new[] { Dict[0].Take(2500).ToArray(), (Dict[1].Take(2500)).ToArray(), (Dict[2].Take(2500)).ToArray(), (Dict[3].Take(2500)).ToArray(), (Dict[4].Take(2500)).ToArray(), (Dict[5].Take(2500)).ToArray(), (Dict[6].Take(2500)).ToArray() }, new[] { 0, 1, 2, 3, 4, 5,6 });
        //for (n = 0; n < 5; n++) LERN(network, new[] { Dict[0].Take(2500).ToArray(), (Dict[1].Take(2500)).ToArray(), (Dict[2].Take(2500)).ToArray(), (Dict[3].Take(2500)).ToArray(), (Dict[4].Take(2500)).ToArray(), (Dict[5].Take(2500)).ToArray(), (Dict[6].Take(2500)).ToArray(), (Dict[7].Take(2500)).ToArray() }, new[] { 0, 1, 2, 3, 4, 5, 6,7 });
        //for (n = 0; n < 6; n++) LERN(network, new[] { Dict[0].Take(2500).ToArray(), (Dict[1].Take(2500)).ToArray(), (Dict[2].Take(2500)).ToArray(), (Dict[3].Take(2500)).ToArray(), (Dict[4].Take(2500)).ToArray(), (Dict[5].Take(2500)).ToArray(), (Dict[6].Take(2500)).ToArray(), (Dict[7].Take(2500)).ToArray(), (Dict[8].Take(2500)).ToArray() }, new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
        //for (n = 0; n < 6; n++) LERN(network, new[] { Dict[0].Take(2500).ToArray(), (Dict[1].Take(2500)).ToArray(), (Dict[2].Take(2500)).ToArray(), (Dict[3].Take(2500)).ToArray(), (Dict[4].Take(2500)).ToArray(), (Dict[5].Take(2500)).ToArray(), (Dict[6].Take(2500)).ToArray(), (Dict[7].Take(2500)).ToArray(), (Dict[8].Take(2500)).ToArray(), (Dict[9].Take(2500)).ToArray() }, new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });








        using var mnist3 = new IdxReader(@"..\..\..\..\Datasets\MNIST\t10k-images.idx3-ubyte");
        using var mnist4 = new IdxReader(@"..\..\..\..\Datasets\MNIST\t10k-labels.idx1-ubyte");

        var images2 = GetSample(mnist3).ToArray();
        var labels2 = GetLabel(mnist4).ToArray();
        Console.Clear();
        Learning.Test(network,images2,labels2);










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

    public static void LERN(Network network,double[][][] trainData,int[] expectedOutput)
    {
        for (int i = 0; i < 5000; i++)
        {
            for (int num = 0; num < trainData.Length; num++)
            {
                Learning.Iterate(network, trainData[num][i], expectedOutput[num]);

            }
            if ((i + 1) % 5 == 0)
            {
                //Thread.Sleep();
                network.GradientDescent(new StochasticGradientDescent(),4);
                Console.WriteLine(network.GetCost());
                Console.WriteLine(trainData.Length);
                var (_, top) = Console.GetCursorPosition();
                Console.SetCursorPosition(0, top - 2);
                network.ResetCost();
            }
        }
    }
}

//network.Layers[0].Weights = DenseMatrix.OfArray(new double[,] { { 0,0,0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } });