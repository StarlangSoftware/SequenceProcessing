package SequenceProcessing.Classification;

import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import ComputationalGraph.Function.Softmax;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.ConcatenatedNode;
import ComputationalGraph.Node.MultiplicationNode;
import Math.*;
import SequenceProcessing.Functions.RemoveBias;
import SequenceProcessing.Functions.Switch;
import SequenceProcessing.Parameters.RecurrentNeuralNetworkParameter;

public class RecurrentNeuralNetworkModel extends ComputationalGraph implements Serializable {

    protected final int wordEmbeddingLength;
    protected ArrayList<Switch> switches;

    public RecurrentNeuralNetworkModel(NeuralNetworkParameter parameter, int wordEmbeddingLength) {
        super(parameter);
        this.wordEmbeddingLength = wordEmbeddingLength;
        this.switches = new ArrayList<>();
    }

    protected ArrayList<Integer> createInputTensors(Tensor instance) {
        ArrayList<Integer> classLabels = new ArrayList<>();
        int timeStep = (instance.getShape()[0] / (wordEmbeddingLength + 1));
        int j = 0;
        for (int i = 0; i < this.inputNodeCount() - 1; i++) {
            if (i < timeStep) {
                this.switches.get(i).setTurn(true);
                ArrayList<Double> values = new ArrayList<>();
                for (int k = 0; k < wordEmbeddingLength; k++) {
                    values.add(instance.getValue(new int[]{j}));
                    j++;
                }
                classLabels.add((int) instance.getValue(new int[]{j}));
                j++;
                this.getInputNode(i).setValue(new Tensor(values, new int[]{1, values.size()}));
            } else {
                this.switches.get(i).setTurn(false);
                ArrayList<Double> values = new ArrayList<>();
                for (int k = 0; k < wordEmbeddingLength; k++) {
                    values.add(0.0);
                    j++;
                }
                classLabels.add(0);
                j++;
                this.getInputNode(i).setValue(new Tensor(values, new int[]{1, values.size()}));
            }
        }
        return classLabels;
    }

    protected void train(ArrayList<Tensor> trainSet, Random random) {
        for (int i = 0; i < parameters.getEpoch(); i++) {
            // System.out.println("Epoch: " + (i + 1));
            // Shuffle
            this.shuffle(trainSet, random);
            for (Tensor instance : trainSet) {
                ArrayList<Integer> classLabels = createInputTensors(instance);
                ArrayList<Double> classLabelValues = new ArrayList<>();
                for (Integer classLabel : classLabels) {
                    for (int in = 0; in < ((RecurrentNeuralNetworkParameter) this.parameters).getClassLabelSize(); in++) {
                        if (in == classLabel) {
                            classLabelValues.add(1.0);
                        } else {
                            classLabelValues.add(0.0);
                        }
                    }
                }
                this.getInputNode(this.inputNodeCount() - 1).setValue(new Tensor(classLabelValues, new int[]{classLabels.size(), ((RecurrentNeuralNetworkParameter) this.parameters).getClassLabelSize()}));
                this.forwardCalculation();
                this.backpropagation();
            }
            parameters.getOptimizer().setLearningRate();
        }
    }

    protected int findTimeStep(ArrayList<Tensor> trainSet) {
        int timeStep = -1;
        for (Tensor tensor : trainSet) {
            int size = tensor.getShape()[0];
            if (timeStep < size / (wordEmbeddingLength + 1)) {
                timeStep = size / (wordEmbeddingLength + 1);
            }
        }
        return timeStep;
    }

    // Many-to-Many RNN
    @Override
    public void train(ArrayList<Tensor> trainSet) {
        Random random = new Random(parameters.getSeed());
        int timeStep = findTimeStep(trainSet);
        ArrayList<ComputationalNode> weights = new ArrayList<>();
        ArrayList<ComputationalNode> recurrentWeights = new ArrayList<>();
        int currentLength = wordEmbeddingLength + 1;
        for (int i = 0; i < ((RecurrentNeuralNetworkParameter) parameters).size(); i++) {
            weights.add(new MultiplicationNode(parameters.initializeWeights(new int[]{currentLength, ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i)}, random)));
            recurrentWeights.add(new MultiplicationNode(parameters.initializeWeights(new int[]{((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i), ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i)}, random)));
            currentLength = ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i) + 1;
        }
        weights.add(new MultiplicationNode(parameters.initializeWeights(new int[]{currentLength, ((RecurrentNeuralNetworkParameter) parameters).getClassLabelSize()}, random)));
        ArrayList<ComputationalNode> currentOldLayers = new ArrayList<>();
        ArrayList<ComputationalNode> outputNodes = new ArrayList<>();
        for (int k = 0; k < timeStep; k++) {
            this.switches.add(new Switch());
            ArrayList<ComputationalNode> newOldLayers = new ArrayList<>();
            ComputationalNode input = new MultiplicationNode(false, true);
            this.addInputNode(input);
            ComputationalNode current = input;
            for (int i = 0; i < ((RecurrentNeuralNetworkParameter) parameters).size(); i++) {
                ComputationalNode aw;
                ComputationalNode aFunction;
                if (!currentOldLayers.isEmpty()) {
                    aw = this.addEdge(current, weights.get(i));
                    ComputationalNode oWithoutBias = this.addEdge(currentOldLayers.get(i), new RemoveBias());
                    ComputationalNode ou = this.addEdge(oWithoutBias, recurrentWeights.get(i));
                    ComputationalNode a = this.addAdditionEdge(aw, ou, false);
                    aFunction = this.addEdge(a, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction(i), true);
                } else {
                    aw = this.addEdge(current, weights.get(i), false);
                    aFunction = this.addEdge(aw, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction(i), true);
                }
                current = aFunction;
                newOldLayers.add(aFunction);
            }
            currentOldLayers = newOldLayers;
            ComputationalNode node = this.addEdge(current, weights.get(weights.size() - 1));
            outputNodes.add(this.addEdge(node, switches.get(k)));
        }
        ConcatenatedNode concatenatedNode = (ConcatenatedNode) this.concatEdges(outputNodes, 0);
        ComputationalNode classLabelNode = this.addLoss(this.addEdge(concatenatedNode, new Softmax()));
        this.addInputNode(classLabelNode);
        train(trainSet, random);
    }

    @Override
    protected ArrayList<Double> getOutputValue(Tensor outputNode) {
        ArrayList<Double> classLabels = new ArrayList<>();
        for (int i = 0; i < outputNode.getShape()[0]; i++) {
            int index = -1;
            double max = Double.MIN_VALUE;
            for (int j = 0; j < outputNode.getShape()[1]; j++) {
                if (max < outputNode.getValue(new int[]{i, j})) {
                    max = outputNode.getValue(new int[]{i, j});
                    index = j;
                }
            }
            classLabels.add((double) index);
        }
        return classLabels;
    }

    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        int count = 0, total = 0;
        for (Tensor instance : testSet) {
            ArrayList<Integer> goldClassLabels = createInputTensors(instance);
            ArrayList<Double> classLabels = this.predict();
            for (int j = 0; j < (instance.getShape()[0] / (wordEmbeddingLength + 1)); j++) {
                if (goldClassLabels.get(j).equals(classLabels.get(j).intValue())) {
                    count++;
                }
                total++;
            }
        }
        return new ClassificationPerformance((count + 0.0) / total);
    }
}
