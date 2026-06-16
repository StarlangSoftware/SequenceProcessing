package SequenceProcessing.Functions;

import java.io.Serializable;

import ComputationalGraph.Function.Function;
import ComputationalGraph.Function.FunctionResults;
import Math.Tensor;

public class RemoveBias implements Function, Serializable {
    @Override
    public FunctionResults calculate(Tensor matrix) {
        double[] data = matrix.getData();
        double[] values = new double[data.length - 1];
        System.arraycopy(data, 0, values, 0, data.length - 1);
        return new FunctionResults(new Tensor(values, new int[]{1, values.length}));
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        double[] data = backward.getData();
        double[] values = new double[data.length + 1];
        System.arraycopy(data, 0, values, 0, data.length);
        values[data.length] = 0.0;
        return new Tensor(values, new int[]{1, values.length});
    }
}
