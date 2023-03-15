package SequenceProcessing.Initializer;

import java.util.Random;
import Math.*;

public class UniformXavier implements Initializer {

    @Override
    public Matrix initialize(int row, int col, Random random) {
        Matrix m = new Matrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                m.setValue(i, j, (2 * random.nextDouble() - 1) * Math.sqrt(6.0 / (row + col)));
            }
        }
        return m;
    }
}
