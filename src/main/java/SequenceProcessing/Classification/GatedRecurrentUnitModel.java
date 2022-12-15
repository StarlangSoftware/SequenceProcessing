package SequenceProcessing.Classification;

import Classification.Parameter.DeepNetworkParameter;
import Corpus.Sentence;
import SequenceProcessing.Sequence.LabelledEmbeddedWord;
import SequenceProcessing.Sequence.SequenceCorpus;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import Math.*;

public class GatedRecurrentUnitModel extends Model implements Serializable {

    public GatedRecurrentUnitModel(SequenceCorpus corpus, DeepNetworkParameter parameters) throws MatrixColumnMismatch, VectorSizeMismatch {
        super(corpus, parameters);
        int epoch = parameters.getEpoch();
        double learningRate = parameters.getLearningRate();
        ArrayList<Vector> aVectors = new ArrayList<>();
        ArrayList<Vector> zVectors = new ArrayList<>();
        ArrayList<Vector> rVectors = new ArrayList<>();
        ArrayList<Matrix> zWeights = new ArrayList<>();
        ArrayList<Matrix> zRecurrentWeights = new ArrayList<>();
        ArrayList<Matrix> rWeights = new ArrayList<>();
        ArrayList<Matrix> rRecurrentWeights = new ArrayList<>();
        for (int i = 0; i < parameters.layerSize(); i++) {
            aVectors.add(new Vector(parameters.getHiddenNodes(i), 0));
            zVectors.add(new Vector(parameters.getHiddenNodes(i), 0));
            rVectors.add(new Vector(parameters.getHiddenNodes(i), 0));
            zWeights.add(new Matrix(this.layers.get(i).size(), this.layers.get(i + 1).size(), -0.01, +0.01, new Random(parameters.getSeed())));
            rWeights.add(new Matrix(this.layers.get(i).size(), this.layers.get(i + 1).size(), -0.01, +0.01, new Random(parameters.getSeed())));
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
                        rVectors.get(l).add(rWeights.get(l).multiplyWithVectorFromRight(layers.get(l)));
                        zVectors.get(l).add(zWeights.get(l).multiplyWithVectorFromRight(layers.get(l)));
                        rVectors.get(l).add(rRecurrentWeights.get(l).multiplyWithVectorFromRight(oldLayers.get(l)));
                        zVectors.get(l).add(zRecurrentWeights.get(l).multiplyWithVectorFromRight(oldLayers.get(l)));
                        switch (parameters.getActivationFunction()) {
                            case RELU:
                                rVectors.get(l).relu();
                                zVectors.get(l).relu();
                                break;
                            case TANH:
                                rVectors.get(l).tanh();
                                zVectors.get(l).tanh();
                                break;
                            case SIGMOID:
                                rVectors.get(l).sigmoid();
                                zVectors.get(l).sigmoid();
                                break;
                            default:
                                break;
                        }
                        aVectors.get(l).add(this.recurrentWeights.get(l).multiplyWithVectorFromRight(rVectors.get(l).elementProduct(oldLayers.get(l))));
                        aVectors.get(l).add(this.weights.get(l).multiplyWithVectorFromRight(layers.get(l)));
                        aVectors.get(l).tanh();
                        layers.get(l + 1).add(calculateOneMinusVector(zVectors.get(l)).elementProduct(oldLayers.get(l)));
                        layers.get(l + 1).add(zVectors.get(l).elementProduct(aVectors.get(l)));
                    }
                    layers.get(layers.size() - 1).add(this.weights.get(this.weights.size() - 1).multiplyWithVectorFromRight(layers.get(layers.size() - 2)));
                    Vector v = normalizeOutput(layers.get(layers.size() - 1));
                    for (int l = 0; l < v.size(); l++) {
                        layers.get(layers.size() - 1).setValue(l, v.getValue(l));
                    }
                    
                    oldLayersUpdate();
                    setLayersValuesToZero();
                    for (int l = 0; l < parameters.layerSize(); l++) {
                        aVectors.get(l).clear();
                        zVectors.get(l).clear();
                        rVectors.get(l).clear();
                    }
                }
                for (Vector oldLayer : this.oldLayers) {
                    oldLayer.clear();
                }
            }
        }
    }
}
