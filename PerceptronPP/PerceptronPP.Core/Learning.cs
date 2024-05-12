namespace PerceptronPP.Core
{
    public static class Learning
    {
        public static void Learn(Network network, int batchSize, double learningCoefficient, double[][] trainData, double[][] expectedOutputs)
        {
            for (var i = 0; i < trainData.Length; i++)
            {
                Iterate(network, trainData[i], expectedOutputs[i]);

                if ((i + 1) % batchSize != 0) continue;
                network.GradientDescend(learningCoefficient);
                Console.WriteLine(network.GetCost());
                Console.WriteLine(i);
                var (_, top) = Console.GetCursorPosition();
                Console.SetCursorPosition(0, top - 2);
                network.ResetCost();
            }

            if (trainData.Length % batchSize != 0) return;
            try { network.GradientDescend(learningCoefficient); }
            catch
            {
                // ignored
            }
        }

        private static void Iterate(Network network, double[] input, double[] expectedOutput)
        {
            var output = network.Compute(input);

            network.BackPropagate(output, expectedOutput);
            network.CalculateCost(output, expectedOutput);
        }

        public static double[] IntToExpectedOutputArray(int value)
        {
            return Enumerable.Range(0, 10).Select((_, i) => i == value ? 1.0 : 0.0).ToArray();
        }
    }
}
