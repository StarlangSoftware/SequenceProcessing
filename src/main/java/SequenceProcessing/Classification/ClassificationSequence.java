package SequenceProcessing.Classification;

import SequenceProcessing.Labeling.Sequence;

public class ClassificationSequence extends Sequence {

    private String classLabel;

    public ClassificationSequence(String classLabel){
        this.classLabel = classLabel;
    }

    public String getClassLabel(){
        return classLabel;
    }
}
