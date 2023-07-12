package SequenceProcessing.Classification;

import Corpus.Sentence;
import SequenceProcessing.Sequence.LabelledVectorizedWord;

import java.io.Serializable;
import java.util.ArrayList;

import Math.*;

public class RecurrentNeuralNetworkModel extends Model implements Serializable {

    @Override
    protected void calculateOutput(Sentence sentence, int index) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        LabelledVectorizedWord word = (LabelledVectorizedWord) sentence.getWord(index);
        createInputVector(word);
        for (int l = 0; l < this.layers.size() - 2; l++) {
            layers.get(l + 1).add(this.recurrentWeights.get(l).multiply(oldLayers.get(l)));
            layers.get(l + 1).add(this.weights.get(l).multiply(this.layers.get(l)));
            layers.set(l + 1, activationFunction(layers.get(l + 1), this.activationFunction));
            layers.set(l + 1, biased(layers.get(l + 1)));
        }
        layers.get(layers.size() - 1).add(this.weights.get(this.weights.size() - 1).multiply(layers.get(layers.size() - 2)));
        normalizeOutput();
    }

    @Override
    protected void backpropagation(Sentence sentence, int index, double learningRate) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        LabelledVectorizedWord word = (LabelledVectorizedWord) sentence.getWord(index);
        Matrix rMinusY = calculateRMinusY(word);
        rMinusY.multiplyWithConstant(learningRate);
        ArrayList<Matrix> deltaWeights = new ArrayList<>();
        ArrayList<Matrix> deltaRecurrentWeights = new ArrayList<>();
        deltaWeights.add(rMinusY.multiply(layers.get(layers.size() - 2).transpose()));
        deltaWeights.add(rMinusY);
        deltaRecurrentWeights.add(rMinusY);
        for (int l = parameters.layerSize() - 1; l >= 0; l--) {
            Matrix delta = deltaWeights.get(deltaWeights.size() - 1).transpose().multiply(weights.get(l + 1).partial(0, weights.get(l + 1).getRow() - 1, 0, weights.get(l + 1).getColumn() - 2)).elementProduct(derivative(layers.get(l + 1).partial(0, layers.get(l + 1).getRow() - 2, 0, layers.get(l + 1).getColumn() - 1), this.activationFunction).transpose()).transpose();
            deltaWeights.set(deltaWeights.size() - 1, delta.multiply(layers.get(l).transpose()));
            deltaRecurrentWeights.set(deltaRecurrentWeights.size() - 1, delta.multiply(oldLayers.get(l).transpose()));
            if (l > 0) {
                deltaWeights.add(delta);
                deltaRecurrentWeights.add(delta);
            }
        }
        weights.get(weights.size() - 1).add(deltaWeights.get(0));
        deltaWeights.remove(0);
        for (int l = 0; l < deltaWeights.size(); l++) {
            weights.get(weights.size() - l - 2).add(deltaWeights.get(l));
            recurrentWeights.get(recurrentWeights.size() - l - 1).add(deltaRecurrentWeights.get(l));
        }
    }
}
