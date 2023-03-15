package SequenceProcessing.Classification;

import Classification.Parameter.ActivationFunction;
import Classification.Parameter.DeepNetworkParameter;
import Corpus.Sentence;
import SequenceProcessing.Initializer.Initializer;
import SequenceProcessing.Sequence.LabelledVectorizedWord;
import SequenceProcessing.Sequence.SequenceCorpus;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import Math.*;

public class GatedRecurrentUnitModel extends Model implements Serializable {

    private ArrayList<Matrix> aVectors;
    private ArrayList<Matrix> zVectors;
    private ArrayList<Matrix> rVectors;
    private ArrayList<Matrix> zWeights;
    private ArrayList<Matrix> zRecurrentWeights;
    private ArrayList<Matrix> rWeights;
    private ArrayList<Matrix> rRecurrentWeights;

    public GatedRecurrentUnitModel(SequenceCorpus corpus, DeepNetworkParameter parameters, Initializer initializer) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        super(corpus, parameters, initializer);
        int epoch = parameters.getEpoch();
        double learningRate = parameters.getLearningRate();
        aVectors = new ArrayList<>();
        zVectors = new ArrayList<>();
        rVectors = new ArrayList<>();
        zWeights = new ArrayList<>();
        zRecurrentWeights = new ArrayList<>();
        rWeights = new ArrayList<>();
        rRecurrentWeights = new ArrayList<>();
        for (int i = 0; i < parameters.layerSize(); i++) {
            aVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            zVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            rVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            zWeights.add(initializer.initialize(this.layers.get(i + 1).getRow(), this.layers.get(i).getRow() + 1, new Random(parameters.getSeed())));
            rWeights.add(initializer.initialize(this.layers.get(i + 1).getRow(), this.layers.get(i).getRow() + 1, new Random(parameters.getSeed())));
            zRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
            rRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
        }
        for (int i = 0; i < epoch; i++) {
            corpus.shuffleSentences(parameters.getSeed());
            for (int j = 0; j < corpus.sentenceCount(); j++) {
                Sentence sentence = corpus.getSentence(j);
                for (int k = 0; k < sentence.wordCount(); k++) {
                    LabelledVectorizedWord word = (LabelledVectorizedWord) sentence.getWord(k);
                    calculateOutput(word);
                    Matrix rMinusY = calculateRMinusY(word);
                    rMinusY.multiplyWithConstant(learningRate);
                    ArrayList<Matrix> deltaWeights = new ArrayList<>();
                    ArrayList<Matrix> deltaRecurrentWeights = new ArrayList<>();
                    ArrayList<Matrix> rDeltaWeights = new ArrayList<>();
                    ArrayList<Matrix> rDeltaRecurrentWeights = new ArrayList<>();
                    ArrayList<Matrix> zDeltaWeights = new ArrayList<>();
                    ArrayList<Matrix> zDeltaRecurrentWeights = new ArrayList<>();
                    deltaWeights.add(rMinusY.multiply(layers.get(layers.size() - 2).transpose()));
                    deltaWeights.add(rMinusY.transpose().multiply(weights.get(weights.size() - 1).partial(0, weights.get(weights.size() - 1).getRow() - 1, 0, weights.get(weights.size() - 1).getColumn() - 2)).transpose());
                    deltaRecurrentWeights.add(deltaWeights.get(deltaWeights.size() - 1).clone());
                    rDeltaWeights.add(deltaWeights.get(deltaWeights.size() - 1).clone());
                    rDeltaRecurrentWeights.add(deltaWeights.get(deltaWeights.size() - 1).clone());
                    zDeltaWeights.add(deltaWeights.get(deltaWeights.size() - 1).clone());
                    zDeltaRecurrentWeights.add(deltaWeights.get(deltaWeights.size() - 1).clone());
                    for (int l = parameters.layerSize() - 1; l >= 0; l--) {
                        Matrix delta = deltaWeights.get(deltaWeights.size() - 1).elementProduct(zVectors.get(l)).elementProduct(derivative(aVectors.get(l), ActivationFunction.TANH));
                        Matrix zDelta = zDeltaWeights.get(zDeltaWeights.size() - 1).elementProduct(aVectors.get(l).difference(oldLayers.get(l))).elementProduct(derivative(zVectors.get(l), this.activationFunction));
                        Matrix rDelta = rDeltaWeights.get(rDeltaWeights.size() - 1).elementProduct(aVectors.get(l).difference(oldLayers.get(l))).elementProduct(derivative(zVectors.get(l), this.activationFunction)).transpose().multiply(recurrentWeights.get(l)).transpose().elementProduct(oldLayers.get(l)).elementProduct(derivative(rVectors.get(l), this.activationFunction));
                        deltaWeights.set(deltaWeights.size() - 1, delta.multiply(layers.get(l).transpose()));
                        deltaRecurrentWeights.set(deltaRecurrentWeights.size() - 1, delta.multiply((rVectors.get(l).elementProduct(oldLayers.get(l))).transpose()));
                        zDeltaWeights.set(zDeltaWeights.size() - 1, zDelta.multiply(layers.get(l).transpose()));
                        zDeltaRecurrentWeights.set(zDeltaRecurrentWeights.size() - 1, zDelta.multiply(oldLayers.get(l).transpose()));
                        rDeltaWeights.set(rDeltaWeights.size() - 1, rDelta.multiply(layers.get(l).transpose()));
                        rDeltaRecurrentWeights.set(rDeltaRecurrentWeights.size() - 1, rDelta.multiply(oldLayers.get(l).transpose()));
                        if (l > 0) {
                            deltaWeights.add(delta.transpose().multiply(weights.get(l).partial(0, weights.get(l).getRow() - 1, 0, weights.get(l).getColumn() - 2)).transpose());
                            deltaRecurrentWeights.add(delta.transpose().multiply(weights.get(l).partial(0, weights.get(l).getRow() - 1, 0, weights.get(l).getColumn() - 2)).transpose());
                            zDeltaWeights.add(zDelta.transpose().multiply(zWeights.get(l).partial(0, zWeights.get(l).getRow() - 1, 0, zWeights.get(l).getColumn() - 2)).transpose());
                            zDeltaRecurrentWeights.add(zDelta.transpose().multiply(zWeights.get(l).partial(0, zWeights.get(l).getRow() - 1, 0, zWeights.get(l).getColumn() - 2)).transpose());
                            rDeltaWeights.add(rDelta.transpose().multiply(rWeights.get(l).partial(0, rWeights.get(l).getRow() - 1, 0, rWeights.get(l).getColumn() - 2)).transpose());
                            rDeltaRecurrentWeights.add(rDelta.transpose().multiply(rWeights.get(l).partial(0, rWeights.get(l).getRow() - 1, 0, rWeights.get(l).getColumn() - 2)).transpose());
                        }
                    }
                    weights.get(weights.size() - 1).add(deltaWeights.get(0));
                    deltaWeights.remove(0);
                    for (int l = 0; l < deltaWeights.size(); l++) {
                        weights.get(weights.size() - l - 2).add(deltaWeights.get(l));
                        rWeights.get(rWeights.size() - l - 1).add(rDeltaWeights.get(l));
                        zWeights.get(zWeights.size() - l - 1).add(zDeltaWeights.get(l));
                        recurrentWeights.get(recurrentWeights.size() - l - 1).add(deltaRecurrentWeights.get(l));
                        zRecurrentWeights.get(zRecurrentWeights.size() - l - 1).add(zDeltaRecurrentWeights.get(l));
                        rRecurrentWeights.get(rRecurrentWeights.size() - l - 1).add(rDeltaRecurrentWeights.get(l));
                    }
                    clear();
                }
                clearOldValues();
            }
            learningRate *= parameters.getEtaDecrease();
        }
    }

    @Override
    protected void calculateOutput(LabelledVectorizedWord word) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        createInputVector(word);
        for (int l = 0; l < this.layers.size() - 2; l++) {
            rVectors.get(l).add(rWeights.get(l).multiply(layers.get(l)));
            zVectors.get(l).add(zWeights.get(l).multiply(layers.get(l)));
            rVectors.get(l).add(rRecurrentWeights.get(l).multiply(oldLayers.get(l)));
            zVectors.get(l).add(zRecurrentWeights.get(l).multiply(oldLayers.get(l)));
            rVectors.set(l, activationFunction(rVectors.get(l), this.activationFunction));
            zVectors.set(l, activationFunction(zVectors.get(l), this.activationFunction));
            aVectors.get(l).add(this.recurrentWeights.get(l).multiply(rVectors.get(l).elementProduct(oldLayers.get(l))));
            aVectors.get(l).add(this.weights.get(l).multiply(layers.get(l)));
            aVectors.set(l, activationFunction(aVectors.get(l), ActivationFunction.TANH));
            layers.get(l + 1).add(calculateOneMinusMatrix(zVectors.get(l)).elementProduct(oldLayers.get(l)));
            layers.get(l + 1).add(zVectors.get(l).elementProduct(aVectors.get(l)));
            layers.set(l + 1, biased(layers.get(l + 1)));
        }
        layers.get(layers.size() - 1).add(this.weights.get(this.weights.size() - 1).multiply(layers.get(layers.size() - 2)));
        normalizeOutput();
    }

    @Override
    protected void clear() {
        super.clear();
        for (int l = 0; l < this.layers.size() - 2; l++) {
            for (int m = 0; m < aVectors.get(l).getRow(); m++) {
                aVectors.get(l).setValue(m, 0, 0.0);
                zVectors.get(l).setValue(m, 0, 0.0);
                rVectors.get(l).setValue(m, 0, 0.0);
            }
        }
    }
}
