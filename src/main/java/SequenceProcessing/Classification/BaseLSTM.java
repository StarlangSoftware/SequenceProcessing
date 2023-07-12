package SequenceProcessing.Classification;

import Classification.Parameter.DeepNetworkParameter;
import SequenceProcessing.Initializer.Initializer;
import SequenceProcessing.Sequence.LabelledVectorizedWord;
import SequenceProcessing.Sequence.SequenceCorpus;
import Math.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public abstract class BaseLSTM extends Model implements Serializable {

    protected ArrayList<Matrix> fVectors;
    protected ArrayList<Matrix> fWeights;
    protected ArrayList<Matrix> fRecurrentWeights;
    protected ArrayList<Matrix> gVectors;
    protected ArrayList<Matrix> gWeights;
    protected ArrayList<Matrix> gRecurrentWeights;
    protected ArrayList<Matrix> iVectors;
    protected ArrayList<Matrix> iWeights;
    protected ArrayList<Matrix> iRecurrentWeights;
    protected ArrayList<Matrix> oVectors;
    protected ArrayList<Matrix> oWeights;
    protected ArrayList<Matrix> oRecurrentWeights;
    protected ArrayList<Matrix> cVectors;
    protected ArrayList<Matrix> cOldVectors;

    public void train(SequenceCorpus corpus, DeepNetworkParameter parameters, Initializer initializer) throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        ArrayList<Integer> layers = new ArrayList<>();
        layers.add(((LabelledVectorizedWord) corpus.getSentence(0).getWord(0)).getVector().size());
        for (int i = 0; i < parameters.layerSize(); i++) {
            layers.add(parameters.getHiddenNodes(i));
        }
        layers.add(corpus.getClassLabels().size());
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
            fWeights.add(initializer.initialize(layers.get(i + 1), layers.get(i) + 1, new Random(parameters.getSeed())));
            gWeights.add(initializer.initialize(layers.get(i + 1), layers.get(i) + 1, new Random(parameters.getSeed())));
            iWeights.add(initializer.initialize(layers.get(i + 1), layers.get(i) + 1, new Random(parameters.getSeed())));
            oWeights.add(initializer.initialize(layers.get(i + 1), layers.get(i) + 1, new Random(parameters.getSeed())));
            fRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
            gRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
            iRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
            oRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
        }
        super.train(corpus, parameters, initializer);
    }

    protected abstract void oldLayersUpdate();

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
