package SequenceProcessing.Functions;

import ComputationalGraph.Function.Function;
import ComputationalGraph.Function.FunctionResults;
import Math.Tensor;

import java.io.Serializable;

public class Variance implements Function, Serializable {

    @Override
    public FunctionResults calculate(Tensor tensor) {
        double[] values = new double[tensor.getShape()[0] * tensor.getShape()[1]];
        double[] variances = new double[tensor.getShape()[0]];
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            double total = 0.0;
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                total += Math.pow(tensor.getValue(new int[]{i, j}), 2);
            }
            variances[i] = total / tensor.getShape()[1];
        }
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                values[i * tensor.getShape()[1] + j] = variances[i];
            }
        }
        return new FunctionResults(new Tensor(values, tensor.getShape()));
    }

    @Override
    public Tensor derivative(Tensor tensor, Tensor backward) {
        double[] values = new double[tensor.getShape()[0] * tensor.getShape()[1]];
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                values[i * tensor.getShape()[1] + j] = 2.0 * Math.sqrt(tensor.getShape()[1] * tensor.getValue(new int[]{i, j})) / tensor.getShape()[1];
            }
        }
        return backward.hadamardProduct(new Tensor(values, tensor.getShape()));
    }
}
