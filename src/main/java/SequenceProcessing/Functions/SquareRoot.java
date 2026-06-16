package SequenceProcessing.Functions;

import ComputationalGraph.Function.Function;
import ComputationalGraph.Function.FunctionResults;
import Math.Tensor;

import java.io.Serializable;

public class SquareRoot implements Function, Serializable {

    private final double epsilon;

    public SquareRoot(double epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public FunctionResults calculate(Tensor tensor) {
        double[] values = new double[tensor.getShape()[0] * tensor.getShape()[1]];
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                values[i * tensor.getShape()[1] + j] = Math.sqrt(this.epsilon + tensor.getValue(new int[]{i, j}));
            }
        }
        return new FunctionResults(new Tensor(values, tensor.getShape()));
    }

    @Override
    public Tensor derivative(Tensor tensor, Tensor backward) {
        double[] values = new double[tensor.getShape()[0] * tensor.getShape()[1]];
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                double val = tensor.getValue(new int[]{i, j});
                values[i * tensor.getShape()[1] + j] = 1.0 / (2.0 * val);
            }
        }
        return backward.hadamardProduct(new Tensor(values, tensor.getShape()));
    }
}
