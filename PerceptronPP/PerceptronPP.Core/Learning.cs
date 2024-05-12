using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptronPP.Core
{
    public class Learning
    {
        public static void Learn(Network network, int batchSize, double learningCoefficient, double[][] trainData, double[][] expectedOutputs)
        {
            double[] output;
            for (int i = 0; i < trainData.Length; i++)
            {
                Iterate(network, trainData[i], expectedOutputs[i]);

                if ((i + 1) % batchSize == 0)
                {
                    //Thread.Sleep();
                    network.GradientDescend(learningCoefficient);
                    Console.WriteLine(network.GetCost());
                    Console.WriteLine(i);
                    var (_, top) = Console.GetCursorPosition();
                    Console.SetCursorPosition(0, top - 2);
                    network.ResetCost();
                }
            }
            if (trainData.Length % batchSize == 0)
                try { network.GradientDescend(learningCoefficient); }
                catch { }
        }

        public static void Test(Network network, double[][] testData, double[][] expectedOutputs)
        {
            Console.Clear();
            int count = 0;
            int correct = 0;
            double[] output;
            for (int i = 0; i < testData.Length; i++)
            {
                output = Iterate(network, testData[i], expectedOutputs[i]);
                if (Check(output, expectedOutputs[i]))
                    correct++;
                count++;

                Console.WriteLine(correct/(double)count * 100);
                Console.WriteLine(i);
                var (_, top) = Console.GetCursorPosition();
                Console.SetCursorPosition(0, top - 2);
            }

        }

        private static bool Check(double[] output, double[] expectedOutput)
        {
            var label = expectedOutput.Select((e,i)=>Tuple.Create(e,i)).Where(tuple => tuple.Item1 == 1).First().Item2;
            var outputLabel = output.Select((e, i) => Tuple.Create(e, i)).OrderByDescending(tuple => tuple.Item1).First().Item2;
            if (label == outputLabel) return true;
            return false;
        }

        public static double[] Iterate(Network network, double[] input, double[] expectedOutput)
        {
            var output = network.Compute(input);
            //Console.WriteLine(String.Join(" ,",output));
            //Console.WriteLine(String.Join(" ,", expectedOutput));
            network.BackPropagate(output,expectedOutput);
            network.CalculateCost(output, expectedOutput);
            
            return output;
        }

        public static double[] IntToExpectedOutputArray(int value)
        {
            return Enumerable.Range(0, 10).Select((e, i) => i == value ? 1.0 : 0.0).ToArray();
        }
    }
}
