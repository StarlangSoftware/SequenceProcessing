package SequenceProcessing.Classification;

import Classification.Parameter.DeepNetworkParameter;
import SequenceProcessing.Sequence.LabelledEmbeddedWord;
import SequenceProcessing.Sequence.SequenceCorpus;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import Math.*;
public abstract class Model implements Serializable {

    protected SequenceCorpus corpus;
    protected ArrayList<Vector> layers;
    protected ArrayList<Vector> oldLayers;
    protected ArrayList<Matrix> weights;
    protected ArrayList<Matrix> recurrentWeights;
    protected ArrayList<String> classLabels;

    public Model(SequenceCorpus corpus, DeepNetworkParameter parameters) {
        this.corpus = corpus;
        ArrayList<Vector> layers = new ArrayList<>();
        ArrayList<Vector> oldLayers = new ArrayList<>();
        ArrayList<Matrix> weights = new ArrayList<>();
        ArrayList<Matrix> recurrentWeights = new ArrayList<>();
        this.classLabels = corpus.getClassLabels();
        int inputSize = ((LabelledEmbeddedWord) corpus.getSentence(0).getWord(0)).getEmbedding().size();
        layers.add(new Vector(inputSize, 0));
        for (int i = 0; i < parameters.layerSize(); i++) {
            oldLayers.add(new Vector(parameters.getHiddenNodes(i), 0));
            layers.add(new Vector(parameters.getHiddenNodes(i), 0));
            recurrentWeights.add(new Matrix(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), -0.01, +0.01, new Random(parameters.getSeed())));
        }
        layers.add(new Vector(classLabels.size(), 0));
        for (int i = 0; i < layers.size() - 1; i++) {
            weights.add(new Matrix(layers.get(i).size(), layers.get(i + 1).size(), -0.01, +0.01, new Random(parameters.getSeed())));
        }
        this.layers = layers;
        this.oldLayers = oldLayers;
        this.weights = weights;
        this.recurrentWeights = recurrentWeights;
    }

    protected void createInputVector(LabelledEmbeddedWord word) {
        for (int i = 0; i < layers.get(0).size(); i++) {
            layers.get(0).setValue(i, word.getEmbedding().getValue(i));
        }
    }

    protected void oldLayersUpdate() {
        for (int i = 0; i < oldLayers.size(); i++) {
            for (int j = 0; j < oldLayers.get(i).size(); j++) {
                oldLayers.get(i).setValue(j, layers.get(i + 1).getValue(j));
            }
        }
    }

    protected void setLayersValuesToZero() {
        for (Vector layer : layers) {
            layer.clear();
        }
    }

    protected Vector calculateOneMinusVector(Vector hidden) throws VectorSizeMismatch {
        Vector one;
        one = new Vector(hidden.size(), 1.0);
        return one.difference(hidden);
    }

    protected Vector normalizeOutput(Vector o) {
        double sum = 0.0;
        double[] values = new double[o.size()];
        for (int i = 0; i < values.length; i++)
            sum += Math.exp(o.getValue(i));
        for (int i = 0; i < values.length; i++)
            values[i] = Math.exp(o.getValue(i)) / sum;
        return new Vector(values);
    }

    public void save(String fileName) {
        FileOutputStream outFile;
        ObjectOutputStream outObject;
        try {
            outFile = new FileOutputStream(fileName);
            outObject = new ObjectOutputStream(outFile);
            outObject.writeObject(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
