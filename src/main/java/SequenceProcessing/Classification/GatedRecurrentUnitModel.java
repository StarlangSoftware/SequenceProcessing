package SequenceProcessing.Classification;

import ComputationalGraph.Function.AdditionByConstant;
import ComputationalGraph.Function.Negation;
import ComputationalGraph.Function.Softmax;
import ComputationalGraph.Function.Tanh;
import ComputationalGraph.NeuralNetworkParameter;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.ConcatenatedNode;
import ComputationalGraph.Node.MultiplicationNode;
import SequenceProcessing.Functions.RemoveBias;
import SequenceProcessing.Functions.Switch;
import Math.Tensor;
import SequenceProcessing.Parameters.RecurrentNeuralNetworkParameter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class GatedRecurrentUnitModel extends RecurrentNeuralNetworkModel implements Serializable {

    public GatedRecurrentUnitModel(NeuralNetworkParameter parameter, int wordEmbeddingLength) {
        super(parameter, wordEmbeddingLength);
        this.switches = new ArrayList<>();
    }

    @Override
    public void train(ArrayList<Tensor> trainSet) {
        Random random = new Random(parameters.getSeed());
        int timeStep = findTimeStep(trainSet);
        ArrayList<ComputationalNode> weights = new ArrayList<>();
        ArrayList<ComputationalNode> recurrentWeights = new ArrayList<>();
        int currentLength = wordEmbeddingLength + 1;
        for (int i = 0; i < ((RecurrentNeuralNetworkParameter) parameters).size(); i++) {
            for (int j = 0; j < 3; j++) {
                weights.add(new MultiplicationNode(parameters.initializeWeights(new int[]{currentLength, ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i)}, random)));
                recurrentWeights.add(new MultiplicationNode(parameters.initializeWeights(new int[]{((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i), ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i)}, random)));
            }
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
                    aw = this.addEdge(current, weights.get((i * 3)));
                    ComputationalNode oWithoutBias = this.addEdge(currentOldLayers.get(i), new RemoveBias());
                    ComputationalNode ou = this.addEdge(oWithoutBias, recurrentWeights.get((i * 3)));
                    ComputationalNode awOu = this.addAdditionEdge(aw, ou, false);
                    ComputationalNode zt = this.addEdge(awOu, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction((i * 2)));
                    aw = this.addEdge(current, weights.get((i * 3) + 1));
                    ou = this.addEdge(oWithoutBias, recurrentWeights.get((i * 3) + 1));
                    awOu = this.addAdditionEdge(aw, ou, false);
                    ComputationalNode rt = this.addEdge(awOu, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction((i * 2) + 1));
                    aw = this.addEdge(current, weights.get((i * 3) + 2));
                    ComputationalNode rtHt1 = this.addEdge(rt, oWithoutBias, false, true);
                    ou = this.addEdge(rtHt1, recurrentWeights.get((i * 3) + 2));
                    awOu = this.addAdditionEdge(aw, ou, false);
                    ComputationalNode hTemp = this.addEdge(awOu, new Tanh());
                    ComputationalNode minusZt = this.addEdge(zt, new Negation());
                    ComputationalNode oneMinusZt = this.addEdge(minusZt, new AdditionByConstant(1.0));
                    aw = this.addEdge(oneMinusZt, oWithoutBias, false, true);
                    ou = this.addEdge(hTemp, zt, false, true);
                    aFunction = this.addAdditionEdge(aw, ou, true);
                } else {
                    aw = this.addEdge(current, weights.get((i * 3)));
                    ComputationalNode zt = this.addEdge(aw, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction((i * 2)));
                    aw = this.addEdge(current, weights.get((i * 3) + 2));
                    ComputationalNode hTemp = this.addEdge(aw, new Tanh());
                    aFunction = this.addEdge(zt, hTemp, true, true);
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
}
