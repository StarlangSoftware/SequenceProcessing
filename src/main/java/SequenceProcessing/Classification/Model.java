package SequenceProcessing.Classification;

import Classification.Parameter.ActivationFunction;
import Classification.Parameter.DeepNetworkParameter;
import Corpus.Sentence;
import SequenceProcessing.Sequence.LabelledVectorizedWord;
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
    protected ArrayList<Matrix> layers;
    protected ArrayList<Matrix> oldLayers;
    protected ArrayList<Matrix> weights;
    protected ArrayList<Matrix> recurrentWeights;
    protected ArrayList<String> classLabels;
    protected ActivationFunction activationFunction;

    public Model(SequenceCorpus corpus, DeepNetworkParameter parameters) {
        this.corpus = corpus;
        this.activationFunction = parameters.getActivationFunction();
        ArrayList<Matrix> layers = new ArrayList<>();
        ArrayList<Matrix> oldLayers = new ArrayList<>();
        ArrayList<Matrix> weights = new ArrayList<>();
        ArrayList<Matrix> recurrentWeights = new ArrayList<>();
        this.classLabels = corpus.getClassLabels();
        int inputSize = ((LabelledVectorizedWord) corpus.getSentence(0).getWord(0)).getVector().size();
        layers.add(new Matrix(inputSize, 1));
        for (int i = 0; i < parameters.layerSize(); i++) {
            oldLayers.add(new Matrix(parameters.getHiddenNodes(i), 1));
            layers.add(new Matrix(parameters.getHiddenNodes(i), 1));
            recurrentWeights.add(new Matrix(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), -0.01, +0.01, new Random(parameters.getSeed())));
        }
        layers.add(new Matrix(classLabels.size(), 1));
        for (int i = 0; i < layers.size() - 1; i++) {
            weights.add(new Matrix(layers.get(i + 1).getRow(), layers.get(i).getRow() + 1, -0.01, +0.01, new Random(parameters.getSeed())));
        }
        this.layers = layers;
        this.oldLayers = oldLayers;
        this.weights = weights;
        this.recurrentWeights = recurrentWeights;
    }

    protected void createInputVector(LabelledVectorizedWord word) {
        for (int i = 0; i < layers.get(0).getRow(); i++) {
            layers.get(0).setValue(i,0, word.getVector().getValue(i));
        }
        layers.set(0, biased(layers.get(0)));
    }

    protected Matrix biased(Matrix m) {
        Matrix v = new Matrix(m.getRow() + 1, m.getColumn());
        for (int i = 0; i < m.getRow(); i++) {
            v.setValue(i, 0, m.getValue(i, 0));
        }
        v.setValue(m.getRow(), 0, 1.0);
        return v;
    }

    protected void oldLayersUpdate() {
        for (int i = 0; i < oldLayers.size(); i++) {
            for (int j = 0; j < oldLayers.get(i).getRow(); j++) {
                oldLayers.get(i).setValue(j, 0, layers.get(i + 1).getValue(j, 0));
            }
        }
    }

    protected void setLayersValuesToZero() {
        for (int j = 0; j < layers.size() - 1; j++) {
            int size = layers.get(j).getRow();
            layers.set(j, new Matrix(size - 1, 1));
            for (int i = 0; i < layers.get(j).getRow(); i++) {
                layers.get(j).setValue(i, 0, 0.0);
            }
        }
        for (int i = 0; i < layers.get(layers.size() - 1).getRow(); i++) {
            layers.get(layers.size() - 1).setValue(i, 0, 0.0);
        }
    }

    protected Matrix calculateOneMinusMatrix(Matrix hidden) {
        Matrix oneMinus = new Matrix(hidden.getRow(), 1);
        for (int i = 0; i < oneMinus.getRow(); i++) {
            oneMinus.setValue(i, 0, 1 - hidden.getValue(i, 0));
        }
        return oneMinus;
    }

    protected void normalizeOutput() {
        double sum = 0.0;
        double[] values = new double[layers.get(layers.size() - 1).getRow()];
        for (int i = 0; i < values.length; i++) {
            sum += Math.exp(layers.get(layers.size() - 1).getValue(i, 0));
        }
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.exp(layers.get(layers.size() - 1).getValue(i, 0)) / sum;
        }
        for (int i = 0; i < values.length; i++) {
            layers.get(layers.size() - 1).setValue(i, 0, values[i]);
        }
    }

    protected Matrix calculateRMinusY(LabelledVectorizedWord word) {
        Matrix r = new Matrix(classLabels.size(), 1);
        int index = classLabels.indexOf(word.getClassLabel());
        r.setValue(index, 0, 1.0);
        for (int i = 0; i < classLabels.size(); i++) {
            r.setValue(i, 0, r.getValue(i, 0) - layers.get(layers.size() - 1).getValue(i, 0));
        }
        return r;
    }

    protected Matrix derivative(Matrix matrix, ActivationFunction function) throws MatrixDimensionMismatch {
        switch (function) {
            case SIGMOID:
            default:
                Matrix oneMinusHidden = calculateOneMinusMatrix(matrix);
                return matrix.elementProduct(oneMinusHidden);
            case TANH:
                Matrix oneMinusA2 = new Matrix(matrix.getRow(), 1);
                Matrix a2 = matrix.elementProduct(matrix);
                for (int i = 0; i < oneMinusA2.getRow(); i++) {
                    oneMinusA2.setValue(i, 0, 1.0 - a2.getValue(i, 0));
                }
                return oneMinusA2;
            case RELU:
                Matrix der = new Matrix(matrix.getRow(), 1);
                for (int i = 0; i < matrix.getRow(); i++) {
                    if (matrix.getValue(i, 0) > 0) {
                        der.setValue(i, 0, 1.0);
                    }
                }
                return der;
        }
    }

    protected void activationFunction(Matrix matrix, ActivationFunction function) {
        switch (function) {
            case SIGMOID:
                for (int i = 0; i < matrix.getRow(); i++) {
                    matrix.setValue(i, 0, 1 / (1 + Math.exp(-matrix.getValue(i, 0))));
                }
                break;
            case RELU:
                for (int i = 0; i < matrix.getRow(); i++) {
                    if (matrix.getValue(i, 0) < 0) {
                        matrix.setValue(i, 0, 0.0);
                    }
                }
                break;
            case TANH:
                for (int i = 0; i < matrix.getRow(); i++) {
                    matrix.setValue(i, 0, Math.tanh(matrix.getValue(i, 0)));
                }
                break;
        }
    }

    protected abstract void calculateOutput(LabelledVectorizedWord word) throws MatrixRowColumnMismatch, MatrixDimensionMismatch;

    protected abstract void clear();

    public ArrayList<String> predict(Sentence sentence) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        ArrayList<String> classLabels = new ArrayList<>();
        for (int i = 0; i < sentence.wordCount(); i++) {
            LabelledVectorizedWord word = (LabelledVectorizedWord) sentence.getWord(i);
            calculateOutput(word);
            double bestValue = Double.MIN_VALUE;
            String best = this.classLabels.get(0);
            for (int j = 0; j < layers.get(layers.size() - 1).getRow(); j++) {
                if (layers.get(layers.size() - 1).getValue(j, 0) > bestValue) {
                    bestValue = layers.get(layers.size() - 1).getValue(j, 0);
                    best = this.classLabels.get(j);
                }
            }
            classLabels.add(best);
            clear();
        }
        for (Matrix oldLayer : this.oldLayers) {
            for (int k = 0; k < oldLayer.getRow(); k++) {
                oldLayer.setValue(k, 0, 0.0);
            }
        }
        return classLabels;
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
