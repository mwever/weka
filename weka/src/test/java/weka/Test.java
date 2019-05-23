package weka;

import java.io.FileReader;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class Test {

	public static void main(final String[] args) throws Exception {
		Classifier c = new J48();
		Instances data = new Instances(new FileReader("../datasets/classification/multi-class/abalone.arff"));
		data.setClassIndex(data.numAttributes() - 1);
		c.buildClassifier(data);
	}

}
