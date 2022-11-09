import SequenceProcessing.Sequence.SequenceCorpus;
import org.junit.Test;

import static org.junit.Assert.*;

public class SequenceCorpusTest {

    @Test
    public void testCorpus01() {
        SequenceCorpus corpus = new SequenceCorpus("disambiguation-penn.txt");
        assertEquals(25957, corpus.sentenceCount());
        assertEquals(264930, corpus.numberOfWords());
    }

    @Test
    public void testCorpus02() {
        SequenceCorpus corpus = new SequenceCorpus("postag-atis-en.txt");
        assertEquals(5432, corpus.sentenceCount());
        assertEquals(61879, corpus.numberOfWords());
    }

    @Test
    public void testCorpus03() {
        SequenceCorpus corpus = new SequenceCorpus("slot-atis-en.txt");
        assertEquals(5432, corpus.sentenceCount());
        assertEquals(61879, corpus.numberOfWords());
    }

    @Test
    public void testCorpus04() {
        SequenceCorpus corpus = new SequenceCorpus("slot-atis-tr.txt");
        assertEquals(5432, corpus.sentenceCount());
        assertEquals(45875, corpus.numberOfWords());
    }

    @Test
    public void testCorpus05() {
        SequenceCorpus corpus = new SequenceCorpus("disambiguation-atis.txt");
        assertEquals(5432, corpus.sentenceCount());
        assertEquals(45875, corpus.numberOfWords());
    }

    @Test
    public void testCorpus06() {
        SequenceCorpus corpus = new SequenceCorpus("metamorpheme-atis.txt");
        assertEquals(5432, corpus.sentenceCount());
        assertEquals(45875, corpus.numberOfWords());
    }

    @Test
    public void testCorpus07() {
        SequenceCorpus corpus = new SequenceCorpus("postag-atis-tr.txt");
        assertEquals(5432, corpus.sentenceCount());
        assertEquals(45875, corpus.numberOfWords());
    }

    @Test
    public void testCorpus08() {
        SequenceCorpus corpus = new SequenceCorpus("metamorpheme-penn.txt");
        assertEquals(25957, corpus.sentenceCount());
        assertEquals(264930, corpus.numberOfWords());
    }

    @Test
    public void testCorpus09() {
        SequenceCorpus corpus = new SequenceCorpus("ner-penn.txt");
        assertEquals(19118, corpus.sentenceCount());
        assertEquals(168654, corpus.numberOfWords());
    }

    @Test
    public void testCorpus10() {
        SequenceCorpus corpus = new SequenceCorpus("postag-penn.txt");
        assertEquals(25957, corpus.sentenceCount());
        assertEquals(264930, corpus.numberOfWords());
    }

    @Test
    public void testCorpus11() {
        SequenceCorpus corpus = new SequenceCorpus("semanticrolelabeling-penn.txt");
        assertEquals(19118, corpus.sentenceCount());
        assertEquals(168654, corpus.numberOfWords());
    }

    @Test
    public void testCorpus12() {
        SequenceCorpus corpus = new SequenceCorpus("semantics-penn.txt");
        assertEquals(19118, corpus.sentenceCount());
        assertEquals(168654, corpus.numberOfWords());
    }

    @Test
    public void testCorpus13() {
        SequenceCorpus corpus = new SequenceCorpus("shallowparse-penn.txt");
        assertEquals(9557, corpus.sentenceCount());
        assertEquals(87279, corpus.numberOfWords());
    }

    @Test
    public void testCorpus14() {
        SequenceCorpus corpus = new SequenceCorpus("disambiguation-tourism.txt");
        assertEquals(19830, corpus.sentenceCount());
        assertEquals(91152, corpus.numberOfWords());
    }

    @Test
    public void testCorpus15() {
        SequenceCorpus corpus = new SequenceCorpus("metamorpheme-tourism.txt");
        assertEquals(19830, corpus.sentenceCount());
        assertEquals(91152, corpus.numberOfWords());
    }

    @Test
    public void testCorpus16() {
        SequenceCorpus corpus = new SequenceCorpus("postag-tourism.txt");
        assertEquals(19830, corpus.sentenceCount());
        assertEquals(91152, corpus.numberOfWords());
    }

    @Test
    public void testCorpus17() {
        SequenceCorpus corpus = new SequenceCorpus("semantics-tourism.txt");
        assertEquals(19830, corpus.sentenceCount());
        assertEquals(91152, corpus.numberOfWords());
    }

    @Test
    public void testCorpus18() {
        SequenceCorpus corpus = new SequenceCorpus("shallowparse-tourism.txt");
        assertEquals(19830, corpus.sentenceCount());
        assertEquals(91152, corpus.numberOfWords());
    }

    @Test
    public void testCorpus19() {
        SequenceCorpus corpus = new SequenceCorpus("disambiguation-kenet.txt");
        assertEquals(18687, corpus.sentenceCount());
        assertEquals(178658, corpus.numberOfWords());
    }

    @Test
    public void testCorpus20() {
        SequenceCorpus corpus = new SequenceCorpus("metamorpheme-kenet.txt");
        assertEquals(18687, corpus.sentenceCount());
        assertEquals(178658, corpus.numberOfWords());
    }

    @Test
    public void testCorpus21() {
        SequenceCorpus corpus = new SequenceCorpus("postag-kenet.txt");
        assertEquals(18687, corpus.sentenceCount());
        assertEquals(178658, corpus.numberOfWords());
    }

    @Test
    public void testCorpus22() {
        SequenceCorpus corpus = new SequenceCorpus("disambiguation-framenet.txt");
        assertEquals(2704, corpus.sentenceCount());
        assertEquals(19286, corpus.numberOfWords());
    }

    @Test
    public void testCorpus23() {
        SequenceCorpus corpus = new SequenceCorpus("metamorpheme-framenet.txt");
        assertEquals(2704, corpus.sentenceCount());
        assertEquals(19286, corpus.numberOfWords());
    }

    @Test
    public void testCorpus24() {
        SequenceCorpus corpus = new SequenceCorpus("postag-framenet.txt");
        assertEquals(2704, corpus.sentenceCount());
        assertEquals(19286, corpus.numberOfWords());
    }

    @Test
    public void testCorpus25() {
        SequenceCorpus corpus = new SequenceCorpus("semanticrolelabeling-framenet.txt");
        assertEquals(2704, corpus.sentenceCount());
        assertEquals(19286, corpus.numberOfWords());
    }

    @Test
    public void testCorpus26() {
        SequenceCorpus corpus = new SequenceCorpus("sentiment-tourism.txt");
        assertEquals(19830, corpus.sentenceCount());
        assertEquals(91152, corpus.numberOfWords());
    }

}
