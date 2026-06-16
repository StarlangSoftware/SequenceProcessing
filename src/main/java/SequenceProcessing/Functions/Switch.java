package SequenceProcessing.Functions;

import java.io.Serializable;

import ComputationalGraph.Function.Function;
import ComputationalGraph.Function.FunctionResults;
import Math.Tensor;

public class Switch implements Function, Serializable {

    private boolean turn;

    public Switch() {
        this.turn = true;
    }

    public void setTurn(boolean turn) {
        this.turn = turn;
    }

    @Override
    public FunctionResults calculate(Tensor matrix) {
        if (this.turn) {
            return new FunctionResults(matrix);
        }
        int size = 1;
        for (int i = 0; i < matrix.getShape().length; i++) {
            size *= matrix.getShape()[i];
        }
        double[] values = new double[size];
        for (int i = 0; i < size; i++) {
            values[i] = 0.0;
        }
        return new FunctionResults(new Tensor(values, matrix.getShape()));
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        if (this.turn) {
            return backward;
        }
        return calculate(value).output();
    }
}
