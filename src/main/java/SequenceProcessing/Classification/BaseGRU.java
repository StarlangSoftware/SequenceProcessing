package SequenceProcessing.Classification;

import Classification.Parameter.DeepNetworkParameter;
import SequenceProcessing.Initializer.Initializer;
import SequenceProcessing.Sequence.LabelledVectorizedWord;
import SequenceProcessing.Sequence.SequenceCorpus;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import Math.*;

public abstract class BaseGRU extends Model implements Serializable {

    protected ArrayList<Matrix> aVectors;
    protected ArrayList<Matrix> zVectors;
    protected ArrayList<Matrix> rVectors;
    protected ArrayList<Matrix> zWeights;
    protected ArrayList<Matrix> zRecurrentWeights;
    protected ArrayList<Matrix> rWeights;
    protected ArrayList<Matrix> rRecurrentWeights;

    public void train(SequenceCorpus corpus, DeepNetworkParameter parameters, Initializer initializer) throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        ArrayList<Integer> layers = new ArrayList<>();
        layers.add(((LabelledVectorizedWord) corpus.getSentence(0).getWord(0)).getVector().size());
        for (int i = 0; i < parameters.layerSize(); i++) {
            layers.add(parameters.getHiddenNodes(i));
        }
        layers.add(corpus.getClassLabels().size());
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
            zWeights.add(initializer.initialize(layers.get(i + 1), layers.get(i) + 1, new Random(parameters.getSeed())));
            rWeights.add(initializer.initialize(layers.get(i + 1), layers.get(i) + 1, new Random(parameters.getSeed())));
            zRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
            rRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
        }
        super.train(corpus, parameters, initializer);
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
