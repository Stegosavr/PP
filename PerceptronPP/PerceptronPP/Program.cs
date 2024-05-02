using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace PerceptronPP
{
    public class Program
    {
        public static void Main()
        {
            var pow = Math.Pow(2, 31);
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            for (var i = 0; i < pow; i++)
            {
                var x = Math.Exp(i + 1);
            }
            stopwatch.Stop();
            Console.WriteLine(stopwatch.Elapsed.ToString());

            // var network = new Perceptron(new[]
            // {
            //     new Layer(3,3),
            //     new Layer(3,0),
            // });
            // network.Layers[0].Input = DenseMatrix.OfArray(new double[,] { { 1, 1, 1 } });
            // //network.Layers[0].Weights = DenseMatrix.OfArray(new double[,] { { 0,0,0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } });
            // network.Randomize();
            // network.ForwardPropagation();
            // var result = network.GetOutput();
            // Console.WriteLine(result);
        }
    }
}
