package SequenceProcessing.Functions;

import ComputationalGraph.Function.Function;
import ComputationalGraph.Function.FunctionResults;
import Math.Tensor;

import java.io.Serializable;

public class Transpose implements Function, Serializable {

    @Override
    public FunctionResults calculate(Tensor tensor) {
        return new FunctionResults(tensor.transpose(new int[]{1, 0}));
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        return backward.transpose(new int[]{1, 0});
    }
}
