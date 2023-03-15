package SequenceProcessing.Initializer;

import Math.*;

public class Random implements Initializer {

    @Override
    public Matrix initialize(int row, int col, java.util.Random random) {
        return new Matrix(row, col, -0.01, +0.01, random);
    }
}
