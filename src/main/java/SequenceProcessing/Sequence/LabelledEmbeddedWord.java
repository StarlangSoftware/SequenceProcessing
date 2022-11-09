package SequenceProcessing.Sequence;

import Math.Vector;

public class LabelledEmbeddedWord extends EmbeddedWord{

    private String classLabel;

    public LabelledEmbeddedWord(String word, Vector embedding, String classLabel) {
        super(word, embedding);
        this.classLabel = classLabel;
    }

    public LabelledEmbeddedWord(String word, String classLabel) {
        super(word);
        this.classLabel = classLabel;
    }

    public String getClassLabel(){
        return classLabel;
    }
}
