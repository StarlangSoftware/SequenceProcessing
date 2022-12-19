package SequenceProcessing.Sequence;

import Dictionary.VectorizedWord;
import Math.Vector;

public class LabelledVectorizedWord extends VectorizedWord {

    private String classLabel;

    public LabelledVectorizedWord(String word, Vector embedding, String classLabel) {
        super(word, embedding);
        this.classLabel = classLabel;
    }

    public LabelledVectorizedWord(String word, String classLabel) {
        super(word, new Vector(300, 0));
        this.classLabel = classLabel;
    }

    public String getClassLabel(){
        return classLabel;
    }
}
