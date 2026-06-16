package SequenceProcessing.Functions;

import ComputationalGraph.Function.Function;
import ComputationalGraph.Function.FunctionResults;
import Math.Tensor;

import java.io.Serializable;

public class Mean implements Function, Serializable {

    @Override
    public FunctionResults calculate(Tensor tensor) {
        double[] values = new double[tensor.getShape()[0] * tensor.getShape()[1]];
        double[] means = new double[tensor.getShape()[0]];
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            double total = 0.0;
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                total += tensor.getValue(new int[]{i, j});
            }
            means[i] = total / tensor.getShape()[1];
        }
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                values[i * tensor.getShape()[1] + j] = means[i];
            }
        }
        return new FunctionResults(new Tensor(values, tensor.getShape()));
    }

    @Override
    public Tensor derivative(Tensor tensor, Tensor backward) {
        double[] values = new double[tensor.getShape()[0] * tensor.getShape()[1]];
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                values[i * tensor.getShape()[1] + j] = 1.0 / tensor.getShape()[1];
            }
        }
        return backward.hadamardProduct(new Tensor(values, tensor.getShape()));
    }
}
