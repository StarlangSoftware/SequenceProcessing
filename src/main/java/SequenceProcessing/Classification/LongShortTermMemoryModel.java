package SequenceProcessing.Classification;

import Classification.Parameter.ActivationFunction;
import Classification.Parameter.DeepNetworkParameter;
import Corpus.Sentence;
import SequenceProcessing.Initializer.Initializer;
import SequenceProcessing.Sequence.LabelledVectorizedWord;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import Math.*;
import SequenceProcessing.Sequence.SequenceCorpus;

public class LongShortTermMemoryModel extends Model implements Serializable {

    private ArrayList<Matrix> fVectors;
    private ArrayList<Matrix> fWeights;
    private ArrayList<Matrix> fRecurrentWeights;
    private ArrayList<Matrix> gVectors;
    private ArrayList<Matrix> gWeights;
    private ArrayList<Matrix> gRecurrentWeights;
    private ArrayList<Matrix> iVectors;
    private ArrayList<Matrix> iWeights;
    private ArrayList<Matrix> iRecurrentWeights;
    private ArrayList<Matrix> oVectors;
    private ArrayList<Matrix> oWeights;
    private ArrayList<Matrix> oRecurrentWeights;
    private ArrayList<Matrix> cVectors;
    private ArrayList<Matrix> cOldVectors;

    public LongShortTermMemoryModel(SequenceCorpus corpus, DeepNetworkParameter parameters, Initializer initializer) throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        super(corpus, parameters, initializer);
        int epoch = parameters.getEpoch();
        double learningRate = parameters.getLearningRate();
        fVectors = new ArrayList<>();
        fWeights = new ArrayList<>();
        fRecurrentWeights = new ArrayList<>();
        gVectors = new ArrayList<>();
        gWeights = new ArrayList<>();
        gRecurrentWeights = new ArrayList<>();
        iVectors = new ArrayList<>();
        iWeights = new ArrayList<>();
        iRecurrentWeights = new ArrayList<>();
        oVectors = new ArrayList<>();
        oWeights = new ArrayList<>();
        oRecurrentWeights = new ArrayList<>();
        cVectors = new ArrayList<>();
        cOldVectors = new ArrayList<>();
        for (int i = 0; i < parameters.layerSize(); i++) {
            fVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            gVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            iVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            oVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            cVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            cOldVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            fWeights.add(initializer.initialize(this.layers.get(i + 1).getRow(), this.layers.get(i).getRow() + 1, new Random(parameters.getSeed())));
            gWeights.add(initializer.initialize(this.layers.get(i + 1).getRow(), this.layers.get(i).getRow() + 1, new Random(parameters.getSeed())));
            iWeights.add(initializer.initialize(this.layers.get(i + 1).getRow(), this.layers.get(i).getRow() + 1, new Random(parameters.getSeed())));
            oWeights.add(initializer.initialize(this.layers.get(i + 1).getRow(), this.layers.get(i).getRow() + 1, new Random(parameters.getSeed())));
            fRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
            gRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
            iRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
            oRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
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
                    Matrix deltaWeight = rMinusY.multiply(layers.get(layers.size() - 2).transpose());
                    ArrayList<Matrix> fDeltaWeights = new ArrayList<>();
                    ArrayList<Matrix> fDeltaRecurrentWeights = new ArrayList<>();
                    ArrayList<Matrix> gDeltaWeights = new ArrayList<>();
                    ArrayList<Matrix> gDeltaRecurrentWeights = new ArrayList<>();
                    ArrayList<Matrix> iDeltaWeights = new ArrayList<>();
                    ArrayList<Matrix> iDeltaRecurrentWeights = new ArrayList<>();
                    ArrayList<Matrix> oDeltaWeights = new ArrayList<>();
                    ArrayList<Matrix> oDeltaRecurrentWeights = new ArrayList<>();
                    fDeltaWeights.add(rMinusY.transpose().multiply(weights.get(weights.size() - 1).partial(0, weights.get(weights.size() - 1).getRow() - 1, 0, weights.get(weights.size() - 1).getColumn() - 2)).transpose());
                    fDeltaRecurrentWeights.add(fDeltaWeights.get(0).clone());
                    gDeltaWeights.add(fDeltaWeights.get(0).clone());
                    gDeltaRecurrentWeights.add(fDeltaWeights.get(0).clone());
                    iDeltaWeights.add(fDeltaWeights.get(0).clone());
                    iDeltaRecurrentWeights.add(fDeltaWeights.get(0).clone());
                    oDeltaWeights.add(fDeltaWeights.get(0).clone());
                    oDeltaRecurrentWeights.add(fDeltaWeights.get(0).clone());
                    for (int l = parameters.layerSize() - 1; l >= 0; l--) {
                        Matrix cTanH = activationFunction(cVectors.get(l), ActivationFunction.TANH);
                        Matrix cDerivative = derivative(cTanH, ActivationFunction.TANH);
                        Matrix fDelta = fDeltaWeights.get(fDeltaWeights.size() - 1).elementProduct(oVectors.get(l).elementProduct(cDerivative)).elementProduct(cOldVectors.get(l)).elementProduct(derivative(fVectors.get(l), activationFunction));
                        Matrix gDelta = gDeltaWeights.get(gDeltaWeights.size() - 1).elementProduct(oVectors.get(l).elementProduct(cDerivative)).elementProduct(iVectors.get(l)).elementProduct(derivative(gVectors.get(l), ActivationFunction.TANH));
                        Matrix iDelta = iDeltaWeights.get(iDeltaWeights.size() - 1).elementProduct(oVectors.get(l).elementProduct(cDerivative)).elementProduct(gVectors.get(l)).elementProduct(derivative(iVectors.get(l), activationFunction));
                        Matrix oDelta = oDeltaWeights.get(oDeltaWeights.size() - 1).elementProduct(cTanH).elementProduct(derivative(oVectors.get(l), activationFunction));
                        fDeltaWeights.set(fDeltaWeights.size() - 1, fDelta.multiply(layers.get(l).transpose()));
                        fDeltaRecurrentWeights.set(fDeltaRecurrentWeights.size() - 1, fDelta.multiply(oldLayers.get(l).transpose()));
                        gDeltaWeights.set(gDeltaWeights.size() - 1, gDelta.multiply(layers.get(l).transpose()));
                        gDeltaRecurrentWeights.set(gDeltaRecurrentWeights.size() - 1, gDelta.multiply(oldLayers.get(l).transpose()));
                        iDeltaWeights.set(iDeltaWeights.size() - 1, iDelta.multiply(layers.get(l).transpose()));
                        iDeltaRecurrentWeights.set(iDeltaRecurrentWeights.size() - 1, iDelta.multiply(oldLayers.get(l).transpose()));
                        oDeltaWeights.set(oDeltaWeights.size() - 1, oDelta.multiply(layers.get(l).transpose()));
                        oDeltaRecurrentWeights.set(oDeltaRecurrentWeights.size() - 1, oDelta.multiply(oldLayers.get(l).transpose()));
                        if (l > 0) {
                            fDeltaWeights.add(fDelta.transpose().multiply(fWeights.get(l).partial(0, fWeights.get(l).getRow() - 1, 0, fWeights.get(l).getColumn() - 2)).transpose());
                            fDeltaRecurrentWeights.add(fDelta.transpose().multiply(fWeights.get(l).partial(0, fWeights.get(l).getRow() - 1, 0, fWeights.get(l).getColumn() - 2)).transpose());
                            gDeltaWeights.add(gDelta.transpose().multiply(gWeights.get(l).partial(0, gWeights.get(l).getRow() - 1, 0, gWeights.get(l).getColumn() - 2)).transpose());
                            gDeltaRecurrentWeights.add(gDelta.transpose().multiply(gWeights.get(l).partial(0, gWeights.get(l).getRow() - 1, 0, gWeights.get(l).getColumn() - 2)).transpose());
                            iDeltaWeights.add(iDelta.transpose().multiply(iWeights.get(l).partial(0, iWeights.get(l).getRow() - 1, 0, iWeights.get(l).getColumn() - 2)).transpose());
                            iDeltaRecurrentWeights.add(iDelta.transpose().multiply(iWeights.get(l).partial(0, iWeights.get(l).getRow() - 1, 0, iWeights.get(l).getColumn() - 2)).transpose());
                            oDeltaWeights.add(oDelta.transpose().multiply(oWeights.get(l).partial(0, oWeights.get(l).getRow() - 1, 0, oWeights.get(l).getColumn() - 2)).transpose());
                            oDeltaRecurrentWeights.add(oDelta.transpose().multiply(oWeights.get(l).partial(0, oWeights.get(l).getRow() - 1, 0, oWeights.get(l).getColumn() - 2)).transpose());
                        }
                    }
                    weights.get(weights.size() - 1).add(deltaWeight);
                    for (int l = 0; l < fDeltaWeights.size(); l++) {
                        fWeights.get(fWeights.size() - l - 1).add(fDeltaWeights.get(l));
                        gWeights.get(gWeights.size() - l - 1).add(gDeltaWeights.get(l));
                        iWeights.get(iWeights.size() - l - 1).add(iDeltaWeights.get(l));
                        oWeights.get(oWeights.size() - l - 1).add(oDeltaWeights.get(l));
                        fRecurrentWeights.get(fRecurrentWeights.size() - l - 1).add(fDeltaRecurrentWeights.get(l));
                        gRecurrentWeights.get(gRecurrentWeights.size() - l - 1).add(gDeltaRecurrentWeights.get(l));
                        iRecurrentWeights.get(iRecurrentWeights.size() - l - 1).add(iDeltaRecurrentWeights.get(l));
                        oRecurrentWeights.get(oRecurrentWeights.size() - l - 1).add(oDeltaRecurrentWeights.get(l));
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
        ArrayList<Matrix> kVectors = new ArrayList<>();
        ArrayList<Matrix> jVectors = new ArrayList<>();
        for (int i = 0; i < this.layers.size() - 2; i++) {
            fVectors.get(i).add(fRecurrentWeights.get(i).multiply(this.oldLayers.get(i)).sum(fWeights.get(i).multiply(this.layers.get(i))));
            fVectors.set(i, activationFunction(fVectors.get(i), this.activationFunction));
            kVectors.add(cOldVectors.get(i).elementProduct(fVectors.get(i)));
            gVectors.get(i).add(gRecurrentWeights.get(i).multiply(this.oldLayers.get(i)).sum(gWeights.get(i).multiply(this.layers.get(i))));
            gVectors.set(i, activationFunction(gVectors.get(i), ActivationFunction.TANH));
            iVectors.get(i).add(iRecurrentWeights.get(i).multiply(this.oldLayers.get(i)).sum(iWeights.get(i).multiply(this.layers.get(i))));
            iVectors.set(i, activationFunction(iVectors.get(i), this.activationFunction));
            jVectors.add(gVectors.get(i).elementProduct(iVectors.get(i)));
            cVectors.get(i).add(jVectors.get(i).sum(kVectors.get(i)));
            oVectors.get(i).add(oRecurrentWeights.get(i).multiply(this.oldLayers.get(i)).sum(oWeights.get(i).multiply(this.layers.get(i))));
            oVectors.set(i, activationFunction(oVectors.get(i), this.activationFunction));
            layers.get(i + 1).add(oVectors.get(i).elementProduct(activationFunction(cVectors.get(i), ActivationFunction.TANH)));
            layers.set(i + 1, biased(layers.get(i + 1)));
        }
        layers.get(layers.size() - 1).add(this.weights.get(this.weights.size() - 1).multiply(layers.get(layers.size() - 2)));
        normalizeOutput();
    }

    protected void oldLayersUpdate() {
        for (int i = 0; i < oldLayers.size(); i++) {
            for (int j = 0; j < oldLayers.get(i).getRow(); j++) {
                oldLayers.get(i).setValue(j, 0, layers.get(i + 1).getValue(j, 0));
                cOldVectors.get(i).setValue(j, 0, cVectors.get(i).getValue(j, 0));
            }
        }
    }

    @Override
    protected void clear() {
        super.clear();
        for (int l = 0; l < this.layers.size() - 2; l++) {
            for (int m = 0; m < fVectors.get(l).getRow(); m++) {
                fVectors.get(l).setValue(m, 0, 0.0);
                gVectors.get(l).setValue(m, 0, 0.0);
                iVectors.get(l).setValue(m, 0, 0.0);
                oVectors.get(l).setValue(m, 0, 0.0);
                cVectors.get(l).setValue(m, 0, 0.0);
            }
        }
    }

    @Override
    protected void clearOldValues() {
        for (int i = 0; i < this.oldLayers.size(); i++) {
            for (int k = 0; k < this.oldLayers.get(i).getRow(); k++) {
                cOldVectors.get(i).setValue(k, 0, 0.0);
                this.oldLayers.get(i).setValue(k, 0, 0.0);
            }
        }
    }
}
