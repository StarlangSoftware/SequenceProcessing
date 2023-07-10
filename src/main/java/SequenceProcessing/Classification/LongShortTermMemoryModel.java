package SequenceProcessing.Classification;

import Classification.Parameter.ActivationFunction;
import Corpus.Sentence;

import java.io.Serializable;
import java.util.ArrayList;

import Math.*;
import SequenceProcessing.Sequence.LabelledVectorizedWord;

public class LongShortTermMemoryModel extends BaseLSTM implements Serializable {

    @Override
    protected void calculateOutput(Sentence sentence, int index) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        LabelledVectorizedWord word = (LabelledVectorizedWord) sentence.getWord(index);
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
    protected void backpropagation(Sentence sentence, int index, double learningRate) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        LabelledVectorizedWord word = (LabelledVectorizedWord) sentence.getWord(index);
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
    }
}
