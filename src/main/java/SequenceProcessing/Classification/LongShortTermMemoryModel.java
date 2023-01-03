package SequenceProcessing.Classification;

import Classification.Parameter.ActivationFunction;
import Classification.Parameter.DeepNetworkParameter;
import Corpus.Sentence;
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

    public LongShortTermMemoryModel(SequenceCorpus corpus, DeepNetworkParameter parameters) throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        super(corpus, parameters);
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
            fWeights.add(new Matrix(this.layers.get(i + 1).getRow(), this.layers.get(i).getRow() + 1, -0.01, +0.01, new Random(parameters.getSeed())));
            gWeights.add(new Matrix(this.layers.get(i + 1).getRow(), this.layers.get(i).getRow() + 1, -0.01, +0.01, new Random(parameters.getSeed())));
            iWeights.add(new Matrix(this.layers.get(i + 1).getRow(), this.layers.get(i).getRow() + 1, -0.01, +0.01, new Random(parameters.getSeed())));
            oWeights.add(new Matrix(this.layers.get(i + 1).getRow(), this.layers.get(i).getRow() + 1, -0.01, +0.01, new Random(parameters.getSeed())));
            fRecurrentWeights.add(new Matrix(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), -0.01, +0.01, new Random(parameters.getSeed())));
            gRecurrentWeights.add(new Matrix(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), -0.01, +0.01, new Random(parameters.getSeed())));
            iRecurrentWeights.add(new Matrix(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), -0.01, +0.01, new Random(parameters.getSeed())));
            oRecurrentWeights.add(new Matrix(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), -0.01, +0.01, new Random(parameters.getSeed())));
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
                    ArrayList<Matrix> fDeltaWeights = new ArrayList<>();
                    ArrayList<Matrix> fDeltaRecurrentWeights = new ArrayList<>();
                    ArrayList<Matrix> gDeltaWeights = new ArrayList<>();
                    ArrayList<Matrix> gDeltaRecurrentWeights = new ArrayList<>();
                    ArrayList<Matrix> iDeltaWeights = new ArrayList<>();
                    ArrayList<Matrix> iDeltaRecurrentWeights = new ArrayList<>();
                    ArrayList<Matrix> oDeltaWeights = new ArrayList<>();
                    ArrayList<Matrix> oDeltaRecurrentWeights = new ArrayList<>();
                    deltaWeights.add(rMinusY.multiply(layers.get(layers.size() - 2).transpose()));
                    deltaWeights.add(rMinusY);
                    deltaRecurrentWeights.add(rMinusY);
                    fDeltaWeights.add(rMinusY);
                    fDeltaRecurrentWeights.add(rMinusY);
                    gDeltaWeights.add(rMinusY);
                    gDeltaRecurrentWeights.add(rMinusY);
                    iDeltaWeights.add(rMinusY);
                    iDeltaRecurrentWeights.add(rMinusY);
                    oDeltaWeights.add(rMinusY);
                    oDeltaRecurrentWeights.add(rMinusY);
                    for (int l = parameters.layerSize() - 1; l >= 0; l--) {

                    }
                    weights.get(weights.size() - 1).add(deltaWeights.get(0));
                    deltaWeights.remove(0);
                    for (int l = 0; l < deltaWeights.size(); l++) {
                        weights.get(weights.size() - l - 2).add(deltaWeights.get(l));
                        fWeights.get(fWeights.size() - l - 1).add(fDeltaWeights.get(l));
                        gWeights.get(gWeights.size() - l - 1).add(gDeltaWeights.get(l));
                        iWeights.get(iWeights.size() - l - 1).add(iDeltaWeights.get(l));
                        oWeights.get(oWeights.size() - l - 1).add(oDeltaWeights.get(l));
                        recurrentWeights.get(recurrentWeights.size() - l - 1).add(deltaRecurrentWeights.get(l));
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
            activationFunction(fVectors.get(i), this.activationFunction);
            kVectors.add(cOldVectors.get(i).elementProduct(fVectors.get(i)));
            gVectors.get(i).add(gRecurrentWeights.get(i).multiply(this.oldLayers.get(i)).sum(gWeights.get(i).multiply(this.layers.get(i))));
            activationFunction(gVectors.get(i), ActivationFunction.TANH);
            iVectors.get(i).add(iRecurrentWeights.get(i).multiply(this.oldLayers.get(i)).sum(iWeights.get(i).multiply(this.layers.get(i))));
            iVectors.set(i, activationFunction(iVectors.get(i), this.activationFunction));
            jVectors.add(gVectors.get(i).elementProduct(iVectors.get(i)));
            cVectors.add(jVectors.get(i).sum(kVectors.get(i)));
            oVectors.get(i).add(oRecurrentWeights.get(i).multiply(this.oldLayers.get(i)).sum(oWeights.get(i).multiply(this.layers.get(i))));
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
