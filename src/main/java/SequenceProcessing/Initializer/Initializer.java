package SequenceProcessing.Initializer;

import Math.*;

import java.util.Random;

public interface Initializer {
    Matrix initialize(int row, int col, Random random);
}
