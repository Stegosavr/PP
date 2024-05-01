using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace PerceptronPP
{
    public class Program
    {
        public static void Main()
        {

            var network = new Perceptron(new[]
            {
                new Layer(3,3),
                new Layer(3,0),
            });
            network.Layers[0].Input = DenseMatrix.OfArray(new double[,] { { 1, 1, 1 } });
            //network.Layers[0].Weights = DenseMatrix.OfArray(new double[,] { { 0,0,0 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } });
            network.Randomize();
            network.ForwardPropagation();
            var result = network.GetOutput();
            Console.WriteLine(result);
        }
    }
}
