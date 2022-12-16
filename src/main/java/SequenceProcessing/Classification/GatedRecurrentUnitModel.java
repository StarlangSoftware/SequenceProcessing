package SequenceProcessing.Classification;

import Classification.Parameter.ActivationFunction;
import Classification.Parameter.DeepNetworkParameter;
import Corpus.Sentence;
import SequenceProcessing.Sequence.LabelledEmbeddedWord;
import SequenceProcessing.Sequence.SequenceCorpus;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import Math.*;

public class GatedRecurrentUnitModel extends Model implements Serializable {

    public GatedRecurrentUnitModel(SequenceCorpus corpus, DeepNetworkParameter parameters) throws MatrixColumnMismatch, VectorSizeMismatch, MatrixRowColumnMismatch, MatrixDimensionMismatch {
        super(corpus, parameters);
        int epoch = parameters.getEpoch();
        double learningRate = parameters.getLearningRate();
        ArrayList<Matrix> aVectors = new ArrayList<>();
        ArrayList<Matrix> zVectors = new ArrayList<>();
        ArrayList<Matrix> rVectors = new ArrayList<>();
        ArrayList<Matrix> zWeights = new ArrayList<>();
        ArrayList<Matrix> zRecurrentWeights = new ArrayList<>();
        ArrayList<Matrix> rWeights = new ArrayList<>();
        ArrayList<Matrix> rRecurrentWeights = new ArrayList<>();
        for (int i = 0; i < parameters.layerSize(); i++) {
            aVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            zVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            rVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            zWeights.add(new Matrix(this.layers.get(i).getRow(), this.layers.get(i + 1).getRow(), -0.01, +0.01, new Random(parameters.getSeed())));
            rWeights.add(new Matrix(this.layers.get(i).getRow(), this.layers.get(i + 1).getRow(), -0.01, +0.01, new Random(parameters.getSeed())));
            zRecurrentWeights.add(new Matrix(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), -0.01, +0.01, new Random(parameters.getSeed())));
            rRecurrentWeights.add(new Matrix(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), -0.01, +0.01, new Random(parameters.getSeed())));
        }
        for (int i = 0; i < epoch; i++) {
            corpus.shuffleSentences(parameters.getSeed());
            for (int j = 0; j < corpus.sentenceCount(); j++) {
                Sentence sentence = corpus.getSentence(j);
                for (int k = 0; k < sentence.wordCount(); k++) {
                    LabelledEmbeddedWord word = (LabelledEmbeddedWord) sentence.getWord(k);
                    createInputVector(word);
                    for (int l = 0; l < this.layers.size() - 2; l++) {
                        rVectors.get(l).add(rWeights.get(l).multiply(layers.get(l)));
                        zVectors.get(l).add(zWeights.get(l).multiply(layers.get(l)));
                        rVectors.get(l).add(rRecurrentWeights.get(l).multiply(oldLayers.get(l)));
                        zVectors.get(l).add(zRecurrentWeights.get(l).multiply(oldLayers.get(l)));
                        activationFunction(rVectors.get(l), parameters.getActivationFunction());
                        activationFunction(zVectors.get(l), parameters.getActivationFunction());
                        aVectors.get(l).add(this.recurrentWeights.get(l).multiply(rVectors.get(l).elementProduct(oldLayers.get(l))));
                        aVectors.get(l).add(this.weights.get(l).multiply(layers.get(l)));
                        activationFunction(aVectors.get(l), ActivationFunction.TANH);
                        layers.get(l + 1).add(calculateOneMinusMatrix(zVectors.get(l)).elementProduct(oldLayers.get(l)));
                        layers.get(l + 1).add(zVectors.get(l).elementProduct(aVectors.get(l)));
                    }
                    layers.get(layers.size() - 1).add(this.weights.get(this.weights.size() - 1).multiply(layers.get(layers.size() - 2)));
                    normalizeOutput();
                    Matrix rMinusY = calculateRMinusY(word);
                    rMinusY.multiplyWithConstant(learningRate);
                    ArrayList<Matrix> deltaWeights = new ArrayList<>();
                    ArrayList<Matrix> rDeltaWeights = new ArrayList<>();
                    ArrayList<Matrix> zDeltaWeights = new ArrayList<>();
                    deltaWeights.add(rMinusY.multiply(layers.get(layers.size() - 2)));
                    deltaWeights.add(rMinusY);
                    rDeltaWeights.add(rMinusY);
                    zDeltaWeights.add(rMinusY);
                    for (int l = parameters.layerSize() - 1; l >= 0; l--) {

                    }

                    oldLayersUpdate();
                    setLayersValuesToZero();
                    for (int l = 0; l < parameters.layerSize(); l++) {
                        for (int m = 0; m < aVectors.get(l).getRow(); m++) {
                            aVectors.get(l).setValue(m, 0, 0.0);
                            zVectors.get(l).setValue(m, 0, 0.0);
                            rVectors.get(l).setValue(m, 0, 0.0);
                        }
                    }
                }
                for (Matrix oldLayer : this.oldLayers) {
                    for (int k = 0; k < oldLayer.getRow(); k++) {
                        oldLayer.setValue(k, 0, 0.0);
                    }
                }
            }
            learningRate *= parameters.getEtaDecrease();
        }
    }
}
